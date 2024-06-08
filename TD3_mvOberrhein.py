import os
import random
import time
from dataclasses import dataclass
from functools import partial

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from RL_sim.mv_oberrhein_env import get_mvob_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "mv-oberrhein"
    """the id of the environment"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e4)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 40
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    temperature: float = 1.1  # TODO
    ensemble_num: int = 3
    ucb_exploration: float = 0.1
    n_envs: int = 5


# def make_env(env_id, seed, idx, capture_video, run_name):
#     def thunk():
#         if capture_video and idx == 0:
#             env = gym.make(env_id, render_mode="rgb_array")
#             env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         else:
#             env = gym.make(env_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         env.action_space.seed(seed)
#         return env
#     return thunk

def make_env(total_timesteps):
    return partial(get_mvob_env, n_timesteps=total_timesteps)


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a,
                ):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class QEnsembles(nn.Module):
    def __init__(self, env,
                 num: int = 2):
        super(QEnsembles, self).__init__()
        self.num = num
        self.ensembles = nn.ModuleList([QNetwork(env) for _ in range(num)])

    def forward(self, x, a,
                ):
        xs = torch.stack([self.ensembles[i](x, a) for i in range(self.num)], dim=0)
        x_mean = xs.mean(dim=0)
        x_std = torch.std(xs, dim=0, keepdim=False)
        return x_mean, x_std  # (B,1),(B,1)


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x), negative_slope=0.3)
        x = self.fc_mu(x)
        # x = torch.tanh(x)
        # x = x * self.action_scale + self.action_bias
        return x


class ActorEnsembles(nn.Module):
    def __init__(self, env, num: int = 2, ):
        super(ActorEnsembles, self).__init__()
        self.num = num
        self.ensembles = nn.ModuleList([Actor(env) for _ in range(num)])
        # action rescaling
        # raise NotImplementedError(str(env.action_space.high)) TODO 把scale改了
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high[0][0].item() - env.action_space.low[0][0].item()) / 2.0,
                         dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high[0][0].item() + env.action_space.low[0][0].item()) / 2.0,
                                        dtype=torch.float32)
        )

    def forward(self, x,  # (Batch,state dim)
                raw: bool = False,
                ):
        xs = torch.stack([self.ensembles[i](x) for i in range(self.num)], dim=0)  # (Ensemble, Batch,action dim)
        if raw:
            return xs  # (Ensemble, Batch,action dim)
        x_mean = xs.mean(dim=0, )  # ( Batch,action dim)
        x_std = torch.std(xs, dim=0, keepdim=False)  # ( Batch,action dim)
        return x_mean, x_std


def diversification(ensemble: nn.Module,
                    state,  # (B,*)
                    actions,  # (B,A)
                    target: torch.Tensor,  # (B,) or (B,A)
                    loss_fn,
                    optimizer: torch.optim.Optimizer,
                    alpha: float = 0.5,
                    regularizer: str = 'GAL',
                    ADP_alpha: float = 0.1,
                    ADP_beta: float = 0.1,

                    ):
    E, B, A = len(ensemble.ensembles), state.shape[0], actions.shape[1]
    assert target.shape[0] == B == actions.shape[0], "batch size"
    assert len(actions.shape) == 2, "actions shape" + str(actions.shape)

    optimizer.zero_grad()

    if regularizer == "ADP":
        # see https://arxiv.org/pdf/1901.08846
        raise NotImplementedError("regularizer")
    elif regularizer == "GAL":
        #  Gradient Alignment Loss (GAL
        # https://arxiv.org/pdf/1901.09981
        x_grads = torch.zeros((E, state.shape[0], state.shape[1] + actions.shape[1])).to(device)
        loss =0
        for id, model in enumerate(ensemble.ensembles):
            state.requires_grad = True
            actions.requires_grad = True
            yi = model(x=state, a=actions)
            loss_i = loss_fn(yi.view(-1), target.view(-1))/B
            grad1 = torch.autograd.grad([loss_i], [state, actions], create_graph=True)
            loss += loss_i
            x_grads[id] = torch.cat(grad1, dim=1).to(device)  # (B,S+A)
        x_grads = x_grads.reshape((E, B, -1))
        x_grads = torch.nn.functional.normalize(x_grads, dim=2, p=2.0)  # (E,B,X)
        cosine_similarity = torch.einsum("ibj,kbj->", x_grads, x_grads) / (E * 2)
        print("lfdsafsda",loss,  cosine_similarity)
        loss += alpha * cosine_similarity
        #TODO grad 的数值太大
        loss.backward()
        # for id, model in enumerate(ensemble.ensembles):
        #     for qqq, jjj in enumerate(model.parameters()):
        #         print("22222222--", id, ",", qqq, jjj.grad.mean())
        optimizer.step()

    elif regularizer == 'DVERGE':
        # https://papers.nips.cc/paper/2020/file/3ad7c2ebb96fcba7cda0cf54a2e802f5-Paper.pdf
        pass
    else:
        raise NotImplementedError("regularizer")


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    envs = gym.vector.SyncVectorEnv([make_env(args.total_timesteps) for _ in range(args.n_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = ActorEnsembles(envs, num=args.ensemble_num).to(device)
    qf1 = QEnsembles(envs, num=args.ensemble_num).to(device)
    qf1_target = QEnsembles(envs, num=args.ensemble_num).to(device)
    target_actor = ActorEnsembles(envs, num=args.ensemble_num).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        n_envs=args.n_envs,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    total_rewards = 0

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(args.n_envs)])
            actions = torch.from_numpy(actions)
        else:
            with torch.no_grad():
                obs_ = torch.from_numpy(obs).to(device)
                actions_raw = actor(obs_, raw=True)  # (Ensemble, Batch,action dim)
                E, B, A = actions_raw.shape
                actions_raw = actions_raw.reshape((E * B, A))  # (Ensemble * Batch,action dim)

                # However, in continuous action spaces, finding the action that maximizes the UCB is not straightforward.
                # To handle this issue, we propose a simple approximation scheme, which first generates N candidate action set from
                # ensemble policies i=1, and then chooses the action that maximizes the UCB. For evaluation, we approximate the maximum a posterior action
                # by averaging the mean of Gaussian distributions modeled by each ensemble policy

                obs_ = obs_.unsqueeze(0).expand(E, -1, -1).reshape((E * B, -1))  # (Ensemble * Batch,state dim)
                q_mean, q_std = qf1(obs_, actions_raw)  # (Ensemble * Batch,1)
                ubc = q_mean + args.ucb_exploration * q_std  # (Ensemble * Batch,1)
                actions_indices = torch.argmax(ubc.reshape((E, B)), dim=0)  # (Batch,)
                actions_indices = actions_indices + (torch.arange(B).to(device) * E)  # (Batch,)
                actions = actions_raw[actions_indices, :]  # (Batch, action dim)

                noise = torch.normal(mean=torch.zeros_like(actions).to(device),
                                     std=torch.ones_like(actions).to(
                                         device) * args.exploration_noise * actor.action_scale.cpu().item())
                actions += noise.to(device)
                # actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        #  actions is torch.Tensor!
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)  # numpy

        total_rewards += rewards[0]
        if global_step % 1000 == 999:
            print("rewards avg at", str(global_step), "is", total_rewards / 1000)
            total_rewards = 0

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        rb.add(obs,
               real_next_obs,
               actions.cpu().numpy(),
               rewards,
               terminations,
               [infos])

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale

                next_state_actions_mean, next_state_actions_std = target_actor(data.next_observations)
                next_state_actions = (next_state_actions_mean + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target_mean, qf1_next_target_std = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    qf1_next_target_mean).view(-1)
                w = torch.sigmoid(qf1_next_target_std.mean(dim=1) * args.temperature) + 0.5  # (Batch,)


            # Weighted Bellman backups ----
            # down-weights the sample transitions with high variance across target Qfunctions, resulting in a loss function
            # for the Q-updates that has a better signal-to-noise ratio.

            def loss_fn(qf1_a_values_mean, next_q_value):
                return (F.mse_loss(qf1_a_values_mean, next_q_value, reduction='none') * w).mean()


            diversification(ensemble=qf1, state=data.observations, actions=data.actions,
                            target=next_q_value, loss_fn=loss_fn,
                            optimizer=q_optimizer, alpha=0.1,
                            regularizer="GAL", )

            if global_step % args.policy_frequency == 0:
                actions_now_mean, actions_now_std = actor(data.observations)
                qf1_mean, qf1_std = qf1(data.observations, actions_now_mean)
                actor_loss = -qf1_mean.mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                # writer.add_scalar("losses/qf1_values", qf1_a_values_mean.mean().item(), global_step)
                # writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.td3_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Actor, QNetwork),
            device=device,
            exploration_noise=args.exploration_noise,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "TD3", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
