import torch
from pandapower import write_to_net
from pandapower.control import TrafoController

from RL_sim.RLControllor import RLControllor

import pandapower.networks as pn


class DiscreteTapRLController(TrafoController):
    def __init__(self, net, tid, side="lv", tol=1e-3, in_service=True, trafotype="2W", level=0, order=0, recycle=True,
                 drop_same_existing_ctrl=False,
                 matching_params=None,
                 **kwargs):
        if matching_params is None:
            matching_params = {"tid": tid, 'trafotype': trafotype}
        super().__init__(net, tid, side, tol=tol, in_service=in_service, level=level, order=order, trafotype=trafotype,
                         drop_same_existing_ctrl=drop_same_existing_ctrl, matching_params=matching_params,
                         **kwargs)
        self.new_action_set = False
        assert net.trafo.at[tid, "tap_max"] >= 0 and net.trafo.at[tid, "tap_min"] <= 0
        self.max_tap = net.trafo.at[tid, "tap_max"]
        self.min_tap = net.trafo.at[tid, "tap_min"]
        self.tap_range = self.max_tap - self.min_tap + 1
        print("tap range of trafo ",tid, " is ",self.tap_range)

    def set_new_action(self, action:torch.Tensor,):
        assert self.new_action_set == False, "act"
        assert action.shape == (self.tap_range,), "discrete"
        action = torch.softmax(action,dim=0)
        action = torch.argmax(action,dim=0).item() - abs(self.min_tap)
        assert self.min_tap <= action <= self.max_tap,"got"+str(action)

        self.tap_pos = action
        self.new_action_set = True

    def is_converged(self, net):
        converged = not self.new_action_set
        return converged

    def control_step(self, net):
        # WRITE TO NET
        write_to_net(net, self.trafotable, self.controlled_tid, 'tap_pos', self.tap_pos, self._read_write_flag)
        self.new_action_set = False


if __name__ == "__main__":
    net = pn.mv_oberrhein()
    print(net.trafo)
    dd = DiscreteTapRLController(net, tid=114)
    print(net.controller)
