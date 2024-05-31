import pandapower as pp
import torch
from pandapower import control, cosphi_from_pq
import pandas as pd
from pandapower import timeseries as ts

# importing a grid from the library
from pandapower.networks import mv_oberrhein


class ShuntCapacitorBanKRLController(control.basic_controller.Controller):
    """
    for pandapower.create.create_shunt_as_capacitor(net, bus, q_mvar, loss_factor, **kwargs)
    """

    def __init__(self, net, gid,
                 data_source=None, p_profile=None, in_service=True,
                 recycle=False, order=0, level=0, **kwargs):
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                         initial_powerflow=True, **kwargs)
        self.gid = gid  # index of the controlled storage
        self.bus = net.shunt.at[gid, "bus"]
        self.p_mw = net.shunt.at[gid, "p_mw"]
        self.q_mvar = net.shunt.at[gid, "q_mvar"]
        self.loss_factor = net.shunt.at[gid, "loss_factor"]
        self.vn_kv = net.shunt.at[gid, "vn_kv"]
        self.step = net.shunt.at[gid, "step"]
        self.max_step = net.shunt.at[gid, "max_step"]

        self.in_service = net.shunt.at[gid, "in_service"]

        self.new_action_set = False

    def set_new_action(self, step):
        assert self.new_action_set == False, "act"
        # assert 0 <= step <= self.max_step
        step = torch.sigmoid(step) * self.max_step * 2
        step = self.max_step if step > self.max_step else step
        self.step = step
        self.new_action_set = True

    def is_converged(self, net):
        converged = ( not self.new_action_set)
        return converged

    def _write_to_net(self, net):
        net.shunt.at[self.gid, "step"] = self.step

    def control_step(self, net):
        self._write_to_net(net)
        self.new_action_set = False


if __name__ == "__main__":
    # importing a grid from the library
    from pandapower.networks import mv_oberrhein, cigre_networks

    # loading the network with the usecase 'generation'
    net = mv_oberrhein()

    print(net)
