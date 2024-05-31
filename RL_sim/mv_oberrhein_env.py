import random
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import torch

from RL_sim.DiscreteTapRLController import DiscreteTapRLController
from RL_sim.GridEnv import GridEnv
from RL_sim.SgenRLController import SgenRLController
from RL_sim.StorageRLController import StorageRLController


def get_mvob_env(n_timesteps=10000, eprice=None):
    net = pn.mv_oberrhein()

    storage_num = 20
    max_e_mwh_all = 3.0
    for i in range(20):
        bus_num = np.random.choice(net.bus.index)
        max_e_mwh = random.random() * max_e_mwh_all
        max_p_mw = random.random() * max_e_mwh
        pp.create_storage(net, bus=bus_num, p_mw=0, max_e_mwh=max_e_mwh, soc_percent=0, max_p_mw=max_p_mw, min_p_mw=0,
                          max_q_mvar=0, min_q_mvar=0, controllable=True, )

    # print(net,"\n","========","\n",)
    print( "控制器有trafo，storage和sgen")
    # 设置controller
    action_dim = 0
    for i in range(net.trafo.shape[0]):
        tid = net.trafo.index[i]
        t_ctrl = DiscreteTapRLController(net=net, tid=tid, )
        action_dim += t_ctrl.tap_range

    for i in range(net.storage.shape[0]):
        gid = net.storage.index[i]
        s_ctrl = StorageRLController(net=net, gid=gid, )
        action_dim += 1
    # print(net.sgen, "\n", "========", "\n", )

    min_power_factor = 0.8
    for i in range(net.sgen.shape[0]):
        gid = net.sgen.index[i]
        p_mw = net.sgen.at[gid, "p_mw"]
        q_mvar = net.sgen.at[gid, "q_mvar"]
        sg_ctrl = SgenRLController(net=net, gid=gid, min_power_factor=min_power_factor,
                                   max_q_mvar=2 * q_mvar, min_q_mvar=0.5 * q_mvar,
                                   max_p_mw=2 * p_mw, min_p_mw=0.5 * p_mw)
        action_dim += 2
    # print(net.controller, "\n", "========", "\n", )
    if not eprice:
        eprice = torch.randn(n_timesteps)
    print("actions_dim  ", action_dim)
    env = GridEnv(net=net, n_timesteps=n_timesteps, a=100, b=72.4, c=0.5,
                  eprice=eprice,
                  constraint_penalty=1, ext_grid_s=1.05, min_power_factor=min_power_factor,
                  action_space_size = action_dim,
                  )



    return env  #, actions_dim


# if __name__ == "__main__":
#     n_timesteps = 10000
#     env = get_mvob_env(n_timesteps)
#     print(env.action_space)
#     state,_ = env.reset()
#     for t in range(n_timesteps):
#         print("step ", t)
#         action = torch.randn(actions_dim).to("cuda")
#         old_state = state.detach().clone()
#         state, reward, done, _ = env.step(action)
#         print("state", old_state - state)
