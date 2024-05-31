import copy
import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pandapower import read_from_net, cosphi_from_pq
from pandapower.control import control_diagnostic, run_control, ControllerNotConverged, prepare_run_ctrl, get_recycle, \
    control_initialization, control_implementation, control_finalization, net_initialization
from pandapower.timeseries.run_time_series import init_time_series, cleanup, print_progress, run_time_step, \
    _call_output_writer, control_time_step, controller_not_converged, pf_not_converged, finalize_step

from RL_sim.DiscreteTapRLController import DiscreteTapRLController
from RL_sim.SgenRLController import SgenRLController
from RL_sim.StorageRLController import StorageRLController


class GridEnv(gym.Env):
    def __init__(self,
                 net,
                 n_timesteps: int,
                 a, b, c,
                 eprice,
                 constraint_penalty,
                 ext_grid_s,
                 action_space_size: int,
                 min_power_factor,
                 max_voltage=1.05,
                 min_voltage=0.95,
                 ):
        super(GridEnv, self).__init__()
        self.a, self.b, self.c = a, b, c
        self.n_timesteps = n_timesteps
        self.eprice = eprice
        self.net = net
        self.initial_net = copy.deepcopy(net)

        # 定义观测空间（observation space）
        bus_num = net.bus.shape[0]
        self.observation_space = spaces.Box(low=-1, high=1, shape=(401,))  # TODO
        # 定义动作空间（action space）
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_space_size,))

        self.ext_grid_s = ext_grid_s
        self.max_voltage = max_voltage
        self.min_voltage = min_voltage
        self.min_power_factor = min_power_factor
        self.constraint_penalty = constraint_penalty

        # 初始化环境状态
        self.state = None
        self.steps_taken = 0

    def reset(self, seed=None,
              continue_on_divergence=False, verbose=True, check_controllers=True,
              run_control_fct=run_control, output_writer_fct=_call_output_writer,
              **kwargs):
        self.run_control_fct = run_control_fct
        self.output_writer_fct = output_writer_fct
        self.kwargs = kwargs
        # 重置环境状态
        self.net = copy.deepcopy(self.initial_net)
        # prepare
        time_steps = range(0, self.n_timesteps)
        ts_variables = init_time_series(self.net, time_steps, continue_on_divergence, verbose, **kwargs)
        # cleanup ppc before first time step
        cleanup(self.net, ts_variables)
        self.ts_variables = ts_variables

        if check_controllers:
            control_diagnostic(self.net)  # produces significant overhead if you run many timeseries of short duration

        self.state = self._get_state()
        self.steps_taken = 0
        print("state", self.state.shape)
        return self.state.cpu(), {}

    def _get_state(self):
        print_progress(self.steps_taken, self.steps_taken, self.ts_variables["time_steps"],
                       self.ts_variables["verbose"],
                       ts_variables=self.ts_variables,
                       **self.kwargs)

        self.ctrl_converged = True
        self.pf_converged = True
        # run time step function for each controller
        control_time_step(self.ts_variables['controller_order'], self.steps_taken)  # const controller goes here
        try:
            # calls controller init, control steps and run function (runpp usually is called in here)
            # run_control_fct(net, ctrl_variables=self.ts_variables, **self.kwargs)
            ctrl_variables = prepare_run_ctrl(self.net, self.ts_variables)
            self.kwargs["recycle"], self.kwargs["only_v_results"] = get_recycle(ctrl_variables)
            self.controller_order = ctrl_variables["controller_order"]
            # initialize each controller prior to the first power flow
            control_initialization(self.controller_order)
            # initial power flow (takes time, but is not needed for every kind of controller)
            self.ctrl_variables = net_initialization(self.net, ctrl_variables, **self.kwargs)

        except ControllerNotConverged:
            self.ctrl_converged = False
            # If controller did not converge do some stuff
            controller_not_converged(self.steps_taken, self.ts_variables)
        except self.ts_variables['errors']:
            # If power flow did not converge simulation aborts or continues if continue_on_divergence is True
            self.pf_converged = False
            pf_not_converged(self.steps_taken, self.ts_variables)
        # get the current state
        res = self.net.res_bus
        p_mw_tensor = torch.tensor(res['p_mw'].values).to("cuda")
        q_mvar_tensor = torch.tensor(res['q_mvar'].values).to("cuda")

        tap_poses = torch.from_numpy(read_from_net(self.net, "trafo", -1, "tap_pos", flag="all_index")).to(
            "cuda")
        tap_poses3w = torch.from_numpy(read_from_net(self.net, "trafo3w", -1, "tap_pos", flag="all_index")).to(
            "cuda")
        storage_soc = torch.from_numpy(
            read_from_net(self.net, "storage", -1, "soc_percent", flag="all_index")).to(
            "cuda")
        storage_max_e_mwh = torch.from_numpy(
            read_from_net(self.net, "storage", -1, "max_e_mwh", flag="all_index")).to(
            "cuda")
        e_price = (self.eprice[self.steps_taken]).clone().detach().to("cuda")

        # for i in [p_mw_tensor, q_mvar_tensor, tap_poses, tap_poses3w, storage_soc, storage_max_e_mwh, e_price]:
        #     print(i.shape,"====")

        e_price = e_price.unsqueeze(0) if len(e_price.shape) == 0 else e_price

        # state 由各节点有功无功，storage 储能，变压器 tap头，此刻电价组成
        if tap_poses3w.shape[0] == 0:
            state = torch.cat(
                (p_mw_tensor, q_mvar_tensor, tap_poses, storage_soc, storage_max_e_mwh, e_price),
                dim=0).to("cuda")
        else:
            state = torch.cat(
                (p_mw_tensor, q_mvar_tensor, tap_poses, tap_poses3w, storage_soc, storage_max_e_mwh, e_price),
                dim=0).to("cuda")
        return state.cpu().detach()

    def _action(self, action:torch.Tensor,
                max_iter=30, ):
        # set the controllers by action
        action_id = 0
        for i, levelorder in enumerate(self.controller_order):
            for cc, net in levelorder:
                if hasattr(cc, 'set_new_action'):
                    if isinstance(cc, SgenRLController):
                        cc.set_new_action(p=action[action_id], q=action[action_id + 1])
                        action_id += 2
                    elif isinstance(cc, DiscreteTapRLController):
                        cc.set_new_action(action=action[action_id:action_id + cc.tap_range])
                        action_id += cc.tap_range
                    elif isinstance(cc, StorageRLController):
                        cc.set_new_action(action[action_id])
                        action_id += 1
                    else:
                        raise NotImplementedError("got unsupported ctrollor type " + str(type(cc)))
        #
        try:
            # run each controller step in given controller order
            control_implementation(self.net, self.controller_order, self.ctrl_variables, max_iter, **self.kwargs)
            # call finalize function of each controller
            control_finalization(self.controller_order)
        except ControllerNotConverged:
            self.ctrl_converged = False
            # If controller did not converge do some stuff
            controller_not_converged(self.steps_taken, self.ts_variables)
        except self.ts_variables['errors']:
            # If power flow did not converge simulation aborts or continues if continue_on_divergence is True
            self.pf_converged = False
            pf_not_converged(self.steps_taken, self.ts_variables)

        self.output_writer_fct(self.net, self.steps_taken, self.pf_converged, self.ctrl_converged, self.ts_variables)
        finalize_step(self.ts_variables['controller_order'], self.steps_taken)

    def step(self, action:torch.Tensor):
        self._action(action)
        self.state = self._get_state()

        # 计算奖励
        reward = self._calculate_reward(constraint_penalty=self.constraint_penalty,
                                        a=self.a, b=self.b, c=self.c,
                                        ext_grid_s=self.ext_grid_s,
                                        max_voltage=self.max_voltage,
                                        min_voltage=self.min_voltage,
                                        min_power_factor=self.min_power_factor, )
        # 增加步数
        self.steps_taken += 1

        # 判断是否达到终止条件
        done = self.steps_taken >= self.n_timesteps

        self.steps_taken += 1
        # 返回观测、奖励、是否终止和额外信息
        # tensor,tensor,tensor,None,None
        return self.state.cpu().detach(), reward.cpu().detach(), torch.BoolTensor([done]).cpu(), {}, {}

    @torch.no_grad()
    def _calculate_reward(self,
                          constraint_penalty,
                          a, b, c,
                          ext_grid_s,
                          max_voltage, min_voltage,
                          min_power_factor,
                          ):
        P_dg = torch.from_numpy(read_from_net(self.net, "sgen", -1, "p_mw", flag="all_index")).to("cuda")
        P_dg = a * (P_dg ** 2) + b * P_dg + c
        P_s = torch.tensor(self.net["res_ext_grid"]["p_mw"].values[0]).to("cuda")
        reward = -P_s - P_dg
        reward = reward.sum()

        # c1 measures the violation of the substation  capacity constraint
        Q_s = torch.tensor(self.net["res_ext_grid"]["q_mvar"].values[0]).to("cuda")
        c1 = torch.sqrt(P_s ** 2 + Q_s ** 2) / ext_grid_s
        c1 = torch.max(torch.zeros_like(c1), c1 - torch.ones_like(c1))
        c1 = c1.sum()

        # c2 reflects the degree of violation of the nodal voltage limits
        vm_pu = torch.from_numpy(self.net["res_bus"]["vm_pu"].values).to("cuda")
        c2_1 = vm_pu - max_voltage
        c2_2 = min_voltage - vm_pu
        c2 = torch.max(torch.zeros_like(c2_1), c2_1) + torch.max(torch.zeros_like(c2_2), c2_2)
        c2 = c2.sum(dim=0)

        # c3 reflects the degree of violation of branch loading limits
        load_percent = torch.from_numpy(self.net["res_line"]["loading_percent"].values).to("cuda") / 100
        c3 = torch.max(torch.zeros_like(load_percent), load_percent - torch.ones_like(load_percent))
        c3 = c3.sum()

        # c4 assesses the violation of the power factor constraint
        P_dg = torch.from_numpy(read_from_net(self.net, "sgen", -1, "p_mw", flag="all_index"))
        Q_dg = torch.from_numpy(read_from_net(self.net, "sgen", -1, "q_mvar", flag="all_index"))
        c4 = torch.from_numpy(cosphi_from_pq(P_dg, Q_dg)[0]).to("cuda")
        c4 = min_power_factor * torch.ones_like(c4) - c4
        c4 = torch.max(torch.zeros_like(c4), c4).to("cuda")
        c4 = c4.sum()

        # c5 violation  of  the  BSS capacity constraint
        min_e = torch.from_numpy(read_from_net(self.net, "storage", -1, "min_e_mwh", flag="all_index")).to(
            "cuda")
        max_e = torch.from_numpy(read_from_net(self.net, "storage", -1, "max_e_mwh", flag="all_index")).to(
            "cuda")
        soc = torch.from_numpy(read_from_net(self.net, "storage", -1, "soc_percent", flag="all_index")).to(
            "cuda")
        c5 = torch.max(torch.zeros_like(soc), soc - torch.ones_like(soc)) + torch.max(torch.zeros_like(soc),
                                                                                      min_e / max_e - soc)
        c5 = c5.sum()

        constraints = c1 + c2 + c3 + c4 + c5

        return (constraint_penalty * constraints - reward).detach()#.item()

    def close(self):
        # cleanup functions after the last time step was calculated
        cleanup(self.net, self.ts_variables)
        # both cleanups, at the start AND at the end, are important!
