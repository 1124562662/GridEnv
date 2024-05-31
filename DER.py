import random
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import torch

from RL_sim.SgenRLController import SgenRLController
from RL_sim.StorageRLController import StorageRLController

from RL_sim.GridEnv import GridEnv
from RL_sim.DiscreteTapRLController import DiscreteTapRLController

# TODO -- 把EV交通网加入，增加不确定性

if __name__ == "__main__":
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

    # 设置controller
    for i in range(net.trafo.shape[0]):
        tid = net.trafo.index[i]
        t_ctrl = DiscreteTapRLController(net=net, tid=tid, )

    for i in range(net.storage.shape[0]):
        gid = net.storage.index[i]
        s_ctrl = StorageRLController(net=net, gid=gid, )
    # print(net.sgen, "\n", "========", "\n", )

    min_power_factor = 0.8
    for i in range(net.sgen.shape[0]):
        gid = net.sgen.index[i]
        p_mw = net.sgen.at[gid, "p_mw"]
        q_mvar = net.sgen.at[gid, "q_mvar"]
        sg_ctrl = SgenRLController(net=net, gid=gid, min_power_factor=min_power_factor,
                                   max_q_mvar=2 * q_mvar, min_q_mvar=0.5 * q_mvar,
                                   max_p_mw=2 * p_mw, min_p_mw=0.5 * p_mw)
    # print(net.controller, "\n", "========", "\n", )
    n_timesteps = 10000
    eprice = torch.randn(n_timesteps)
    env = GridEnv(net=net, n_timesteps=n_timesteps, a=100, b=72.4, c=0.5,
                  eprice=eprice,
                  constraint_penalty=1, ext_grid_s=10000, min_power_factor=min_power_factor,
                  )
    state = env.reset()
    actions_dim = net.controller.shape[0]
    for cid, c in enumerate(net.controller["object"]):
        if isinstance(c, SgenRLController):
            actions_dim += 1
    assert actions_dim == net.controller.shape[0] + net.sgen.shape[0]

    for t in range(n_timesteps):
        print("step ",t)
        action = torch.randn(actions_dim).to("cuda")
        # action_id = 0
        # for cid, c in enumerate(net.controller["object"]):
        #     if isinstance(c, SgenRLController):
        #         c.set_new_action(p=actions[action_id], q=actions[action_id + 1])
        #         action_id += 2
        #     else:
        #         print(type(c),action_id,actions[action_id])
        #         c.set_new_action(actions[action_id])
        #         action_id += 1
        old_state = state.detach().clone()
        state,reward,done,_= env.step(action)
        print("state",old_state - state)

