import pandapower as pp
import torch
from pandapower import control, cosphi_from_pq
import pandas as pd
from pandapower import timeseries as ts

# importing a grid from the library
from pandapower.networks import mv_oberrhein

class SgenRLController(control.basic_controller.Controller):
    """
        PQ node
    """
    def __init__(self, net, gid,min_power_factor,
                 min_q_mvar = None, max_q_mvar =None,max_p_mw=None,min_p_mw=None,
                 data_source=None, p_profile=None, in_service=True,
                 recycle=False, order=0, level=0, **kwargs):
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                         initial_powerflow=True, **kwargs)
        self.gid = gid  # index of the controlled storage
        self.bus = net.sgen.at[gid, "bus"]
        self.p_mw = net.sgen.at[gid, "p_mw"]
        self.q_mvar = net.sgen.at[gid, "q_mvar"]
        self.sn_mva = net.sgen.at[gid, "sn_mva"]
        self.name = net.sgen.at[gid, "name"]
        self.gen_type = net.sgen.at[gid, "type"]
        self.in_service = net.sgen.at[gid, "in_service"]
        # self.slack_weight = net.sgen.at[gid,"slack_weight"]
        # self.slack = net.sgen.at[gid,"slack"]
        self.scaling = net.sgen.at[gid,"scaling"]

        self.min_q_mvar = min_q_mvar
        self.max_q_mvar = max_q_mvar # if net.sgen.at[gid,"max_q_mvar"] else  net.sgen.at[gid,"max_q_mvar"]
        self.max_p_mw = max_p_mw # if net.sgen.at[gid,"max_p_mw"] else net.sgen.at[gid,"max_p_mw"]
        self.min_p_mw = min_p_mw # if net.sgen.at[gid,"min_p_mw"] else net.sgen.at[gid,"min_p_mw"]
        #
        self.min_power_factor = min_power_factor
        self.new_action_set = False

    def _check_power_factor(self,p,q):
        # min_pf < pf < 1
        cos_phi = cosphi_from_pq(p,q)
        return   self.min_power_factor <= cos_phi[0] <= 1

    def set_new_action(self,
                       p:torch.Tensor,
                       q:torch.Tensor,):
        assert self.new_action_set == False, "act"
        # assert self.min_p_mw <= p <= self.max_p_mw
        p = (torch.sigmoid(p) - 0.5) * 2.5 * self.max_p_mw
        p,q = p.item(),q.item()
        if p > self.max_p_mw:
            p = self.max_p_mw
        elif p < self.min_p_mw:
            p = -self.min_p_mw

        # assert self._check_power_factor(p,q)
        if not self._check_power_factor(p,q):
            # print("power factor not satisfied")
            pass
        self.p_mw = p
        self.q_mvar = q
        self.new_action_set = True

    def is_converged(self, net):
        converged = not self.new_action_set
        return converged

    def _write_to_net(self, net):
        net.sgen.at[self.gid, "p_mw"] = self.p_mw
        net.sgen.at[self.gid, "q_mvar"] = self.q_mvar

    def control_step(self, net):
        self._write_to_net(net)
        self.new_action_set = False

if __name__ == "__main__":
    # importing a grid from the library
    from pandapower.networks import mv_oberrhein,cigre_networks

    # loading the network with the usecase 'generation'
    net = mv_oberrhein()
    print(net)