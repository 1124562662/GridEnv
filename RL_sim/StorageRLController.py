import pandapower as pp
import torch
from pandapower import control
import pandas as pd
from pandapower import timeseries as ts

# importing a grid from the library
from pandapower.networks import mv_oberrhein


class StorageRLController(control.basic_controller.Controller):
    """
        Example class of a Storage-Controller. Models an abstract energy storage.
    """

    def __init__(self, net, gid, data_source=None, p_profile=None, in_service=True,
                 recycle=False, order=0, level=0, **kwargs):
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                         initial_powerflow=True, **kwargs)

        # read generator attributes from net
        self.gid = gid  # index of the controlled storage
        self.bus = net.storage.at[gid, "bus"]
        self.p_mw = net.storage.at[gid, "p_mw"]
        self.q_mvar = net.storage.at[gid, "q_mvar"]
        # self.sn_mva = net.storage.at[gid, "sn_mva"]
        self.name = net.storage.at[gid, "name"]
        self.gen_type = net.storage.at[gid, "type"]
        self.in_service = net.storage.at[gid, "in_service"]

        # specific attributes
        self.max_e_mwh = net.storage.at[gid, "max_e_mwh"]
        self.soc_percent = net.storage.at[gid, "soc_percent"]

        self.max_p_mw = net.storage.at[gid, "max_p_mw"]

        # profile attributes
        # self.data_source = data_source
        # self.p_profile = p_profile

        self.new_action_set = False

    def set_new_action(self, action):
        assert self.new_action_set == False, "act"
        # assert -self.max_p_mw <= action <= self.max_p_mw
        action = (torch.sigmoid(action) - 0.5) * 2.5* self.max_p_mw
        action = action.item()
        if action > self.max_p_mw:
            action = self.max_p_mw
        elif action < -self.max_p_mw:
            action = -self.max_p_mw

        self.p_mw = action
        self.new_action_set = True

    # We choose to represent the storage-unit as a storage element in pandapower.
    # We start with a function calculating the amout of stored energy:
    def get_stored_ernergy(self):
        # calculating the stored energy
        return self.max_e_mwh * self.soc_percent / 100

    def is_converged(self, net):
        converged = not   self.new_action_set
        return converged

    # Also a first step we want our controller to be able to write its P and Q and state of charge values back to the
    # data structure net.
    def write_to_net(self, net):
        # write p, q and soc_percent to bus within the net
        net.storage.at[self.gid, "p_mw"] = self.p_mw
        net.storage.at[self.gid, "q_mvar"] = self.q_mvar
        net.storage.at[self.gid, "soc_percent"] = self.soc_percent

    def control_step(self, net):
        self.soc_percent += (self.p_mw * (1) * 15 / 60) / self.max_e_mwh * 100

        # constraints
        self.soc_percent = 1 if self.soc_percent > 1 else self.soc_percent
        self.soc_percent = 0 if self.soc_percent < 0 else self.soc_percent
        self.write_to_net(net)
        self.new_action_set = False


if __name__ == "__main__":
    # importing a grid from the library
    from pandapower.networks import mv_oberrhein

    # loading the network with the usecase 'generation'
    net = mv_oberrhein()
    pp.runpp(net)

    # creating a simple time series
    framedata = pd.DataFrame([0.1, .05, 0.1, .005, -0.2, 0], columns=['P'])
    datasource = ts.DFData(framedata)

    # creating storage unit in the grid, which will be controlled by our controller
    store_el = pp.create_storage(net, 30, p_mw=.1, q_mvar=0, max_e_mwh=0.1, )

    # creating an Object of our new build storage controller, controlling the storage unit
    # ctrl = StorageRLController(net=net, gid=store_el, data_source=datasource, p_profile='P')
