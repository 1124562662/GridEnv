import pandapower as pp
import torch
from pandapower import control
import pandas as pd
from pandapower import timeseries as ts

# importing a grid from the library
from pandapower.networks import mv_oberrhein

class GenRLController(control.basic_controller.Controller):
    """
        PV node, where V holds fixed.
    """
    def __init__(self, net, gid,
                 data_source=None, p_profile=None, in_service=True,
                 recycle=False, order=0, level=0, **kwargs):
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                         initial_powerflow=True, **kwargs)
        self.gid = gid  # index of the controlled storage
        self.bus = net.gen.at[gid, "bus"]
        self.p_mw = net.gen.at[gid, "p_mw"]
        self.vm_pu = net.gen.at[gid, "vm_pu"]
        self.sn_mva = net.gen.at[gid, "sn_mva"]
        self.name = net.gen.at[gid, "name"]
        self.gen_type = net.gen.at[gid, "type"]
        self.in_service = net.gen.at[gid, "in_service"]
        self.slack_weight = net.gen.at[gid,"slack_weight"]
        self.slack = net.gen.at[gid,"slack"]
        self.scaling = net.gen.at[gid,"scaling"]
        self.min_q_mvar = net.gen.at[gid,"min_q_mvar"]
        self.max_q_mvar = net.gen.at[gid,"max_q_mvar"]
        self.max_p_mw = net.gen.at[gid,"max_p_mw"]
        self.min_p_mw = net.gen.at[gid,"min_p_mw"]
        #
        # self.min_power_factor = min_power_factor
        self.new_action_set = False

    def _check_power_factor(self,p):
        # min_pf < pf < 1
        return True

    def set_new_action(self, p :torch.Tensor,):
        assert self.new_action_set == False, "act"
        assert self.min_p_mw <= p <= self.max_p_mw
        assert self._check_power_factor(p)
        self.p_mw = p
        self.new_action_set = True

    def is_converged(self, net):
        converged = ( not self.new_action_set)
        return converged

    def _write_to_net(self, net):
        net.gen.at[self.gid, "p_mw"] = self.p_mw
        # net.storage.at[self.gid, "q_mvar"] = self.q_mvar
        # net.storage.at[self.gid, "soc_percent"] = self.soc_percent

    def control_step(self, net):
        self._write_to_net(net)
        self.new_action_set = False

if __name__ == "__main__":
    # importing a grid from the library
    from pandapower.networks import mv_oberrhein,cigre_networks

    # loading the network with the usecase 'generation'
    net = mv_oberrhein()
    print(net)