import pandapower as pp
from pandapower import control
import pandas as pd
from pandapower import timeseries as ts

# importing a grid from the library
from pandapower.networks import mv_oberrhein
import numpy as np

from pandapower.auxiliary import read_from_net, write_to_net
from pandapower.control.controller.trafo_control import TrafoController
import gymnasium as gym
import os
import numpy as np
import pandas as pd
import tempfile
import random
import pandapower as pp
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl

import matplotlib.pyplot as plt

random.seed(10)







class DiscreteTapControl(TrafoController):
    """
    Trafo Controller with local tap changer voltage control.

    INPUT:
        **net** (attrdict) - Pandapower struct

        **tid** (int) - ID of the trafo that is controlled

    OPTIONAL:

        **side** (string, "lv") - Side of the transformer where the voltage is controlled (hv or lv)

        **trafotype** (float, "2W") - Trafo type ("2W" or "3W")

        **tol** (float, 0.001) - Voltage tolerance band at bus in Percent (default: 1% = 0.01pu)

        **in_service** (bool, True) - Indicates if the controller is currently in_service

        **drop_same_existing_ctrl** (bool, False) - Indicates if already existing controllers of the same type and with the same matching parameters (e.g. at same element) should be dropped
    """

    def __init__(self, net, tid,  side="lv", trafotype="2W",
                 tol=1e-3, in_service=True, level=0, order=0, drop_same_existing_ctrl=False,
                 matching_params=None, **kwargs):
        if matching_params is None:
            matching_params = {"tid": tid, 'trafotype': trafotype}
        super().__init__(net, tid, side, tol=tol, in_service=in_service, level=level, order=order, trafotype=trafotype,
                         drop_same_existing_ctrl=drop_same_existing_ctrl, matching_params=matching_params,
                         **kwargs)

        #self.vm_lower_pu = vm_lower_pu
        #self.vm_upper_pu = vm_upper_pu

        self.vm_delta_pu = self.tap_step_percent / 100. * .5 + self.tol
        self.vm_set_pu = kwargs.get("vm_set_pu")

    @classmethod
    def from_tap_step_percent(cls, net, tid, vm_set_pu, side="lv", trafotype="2W", tol=1e-3, in_service=True, order=0,
                              drop_same_existing_ctrl=False, matching_params=None, **kwargs):
        """
        Alternative mode of the controller, which uses a set point for voltage and the value of net.trafo.tap_step_percent to calculate
        vm_upper_pu and vm_lower_pu. To this end, the parameter vm_set_pu should be provided, instead of vm_lower_pu and vm_upper_pu.
        To use this mode of the controller, the controller can be initialized as following:

        c = DiscreteTapControl.from_tap_step_percent(net, tid, vm_set_pu)

        INPUT:
            **net** (attrdict) - Pandapower struct

            **tid** (int) - ID of the trafo that is controlled

            **vm_set_pu** (float) - Voltage setpoint in pu
        """
        self = cls(net, tid=tid, vm_lower_pu=None, vm_upper_pu=None, side=side, trafotype=trafotype, tol=tol,
                   in_service=in_service, order=order, drop_same_existing_ctrl=drop_same_existing_ctrl,
                   matching_params=matching_params, vm_set_pu=vm_set_pu, **kwargs)
        return self

    @property
    def vm_set_pu(self):
        return self._vm_set_pu

    @vm_set_pu.setter
    def vm_set_pu(self, value):
        self._vm_set_pu = value
        if value is None:
            return
        self.vm_lower_pu = value - self.vm_delta_pu
        self.vm_upper_pu = value + self.vm_delta_pu

    def initialize_control(self, net):
        super().initialize_control(net)
        if hasattr(self, 'vm_set_pu') and self.vm_set_pu is not None:
            self.vm_delta_pu = self.tap_step_percent / 100. * .5 + self.tol

    def control_step(self, net):
        """
        Implements one step of the Discrete controller, always stepping only one tap position up or down
        """
        if self.nothing_to_do(net):
            return

        vm_pu = read_from_net(net, "res_bus", self.controlled_bus, "vm_pu", self._read_write_flag)
        self.tap_pos = read_from_net(net, self.trafotable, self.controlled_tid, "tap_pos", self._read_write_flag)

        increment = np.where(self.tap_side_coeff * self.tap_sign == 1,
                             np.where(np.logical_and(vm_pu < self.vm_lower_pu, self.tap_pos > self.tap_min), -1,
                                      np.where(np.logical_and(vm_pu > self.vm_upper_pu, self.tap_pos < self.tap_max), 1, 0)),
                             np.where(np.logical_and(vm_pu < self.vm_lower_pu, self.tap_pos < self.tap_max), 1,
                                      np.where(np.logical_and(vm_pu > self.vm_upper_pu, self.tap_pos > self.tap_min), -1, 0)))

        self.tap_pos += increment

        # WRITE TO NET
        write_to_net(net, self.trafotable, self.controlled_tid, 'tap_pos', self.tap_pos, self._read_write_flag)

    def is_converged(self, net):
        """
        Checks if the voltage is within the desired voltage band, then returns True
        """
        if self.nothing_to_do(net):
            return True

        vm_pu = read_from_net(net, "res_bus", self.controlled_bus, "vm_pu", self._read_write_flag)
        # this is possible in case the trafo is set out of service by the connectivity check
        is_nan = np.isnan(vm_pu)
        self.tap_pos = read_from_net(net, self.trafotable, self.controlled_tid, "tap_pos", self._read_write_flag)

        reached_limit = np.where(self.tap_side_coeff * self.tap_sign == 1,
                                 (vm_pu < self.vm_lower_pu) & (self.tap_pos == self.tap_min) |
                                 (vm_pu > self.vm_upper_pu) & (self.tap_pos == self.tap_max),
                                 (vm_pu < self.vm_lower_pu) & (self.tap_pos == self.tap_max) |
                                 (vm_pu > self.vm_upper_pu) & (self.tap_pos == self.tap_min))

        converged = np.logical_or(reached_limit, np.logical_and(self.vm_lower_pu < vm_pu, vm_pu < self.vm_upper_pu))

        return np.all(np.logical_or(converged, is_nan))


class Storage(control.basic_controller.Controller):
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
        self.sn_mva = net.storage.at[gid, "sn_mva"]
        self.name = net.storage.at[gid, "name"]
        self.gen_type = net.storage.at[gid, "type"]
        self.in_service = net.storage.at[gid, "in_service"]
        self.applied = False

        # specific attributes
        self.max_e_mwh = net.storage.at[gid, "max_e_mwh"]
        self.soc_percent = net.storage.at[gid, "soc_percent"] = 0

        # profile attributes
        self.data_source = data_source
        self.p_profile = p_profile
        self.last_time_step = None

    # We choose to represent the storage-unit as a storage element in pandapower.
    # We start with a function calculating the amout of stored energy:
    def get_stored_ernergy(self):
        # calculating the stored energy
        return self.max_e_mwh * self.soc_percent / 100

        # convergence check

    # Also remember that 'is_converged()' returns the boolean value of convergence:
    def is_converged(self, net):
        # check if controller already was applied
        return self.applied

    # Also a first step we want our controller to be able to write its P and Q and state of charge values back to the
    # data structure net.
    def write_to_net(self, net):
        # write p, q and soc_percent to bus within the net
        net.storage.at[self.gid, "p_mw"] = self.p_mw
        net.storage.at[self.gid, "q_mvar"] = self.q_mvar
        net.storage.at[self.gid, "soc_percent"] = self.soc_percent

    # In case the controller is not yet converged, the control step is executed. In the example it simply
    # adopts a new value according to the previously calculated target and writes back to the net.
    def control_step(self, net):
        # Call write_to_net and set the applied variable True
        self.write_to_net(net)
        self.applied = True

    # In a time-series simulation the battery should read new power values from a profile and keep track
    # of its state of charge as depicted below.
    def time_step(self, net, time):
        # keep track of the soc (assuming time is given in 15min values)
        if self.last_time_step is not None:
            # The amount of Energy produce or consumed in the last timestep is added relative to the
            # maximum of the possible stored energy
            self.soc_percent += (self.p_mw * (time - self.last_time_step) * 15 / 60) / self.max_e_mwh * 100
        self.last_time_step = time

        # read new values from a profile
        if self.data_source:
            if self.p_profile is not None:
                self.p_mw = self.data_source.get_time_step_value(time_step=time,
                                                                 profile_name=self.p_profile)

        self.applied = False  # reset applied variableclass Storage(control.basic_controller.Controller):


def simple_test_net():
    """
    simple net that looks like:

    external_grid b0---b1 trasformer(110/20) b2----b3 load
                                     |
                                     |
                            b4 static generation
    """
    net = pp.create_empty_network()
    pp.set_user_pf_options(net, init_vm_pu = "flat", init_va_degree = "dc", calculate_voltage_angles=True)

    b0 = pp.create_bus(net, 110)
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 20)
    b3 = pp.create_bus(net, 20)
    b4 = pp.create_bus(net, 20)

    pp.create_ext_grid(net, b0)
    pp.create_line(net, b0, b1, 10, "149-AL1/24-ST1A 110.0")


    # [-9,9]
    pp.create_transformer(net, b1, b2, "25 MVA 110/20 kV", name='tr1')
    pp.create_line(net, b2, b3, 10, "184-AL1/30-ST1A 20.0")
    pp.create_line(net, b2, b4, 10, "184-AL1/30-ST1A 20.0")

    pp.create_load(net, b3, p_mw=15., q_mvar=10., name='load1')
    pp.create_sgen(net, b4, p_mw=20., q_mvar=0.15, name='sgen1')
    return net


def create_data_source(n_timesteps=24):
    profiles = pd.DataFrame()

    profiles['load1_p'] = np.random.random(n_timesteps) * 15.
    profiles['sgen1_p'] = np.random.random(n_timesteps) * 20.

    ds = DFData(profiles)
    return profiles, ds

def create_controllers(net, ds):
    ConstControl(net, element='load', variable='p_mw', element_index=[0],
                 data_source=ds, profile_name=["load1_p"])
    ConstControl(net, element='sgen', variable='p_mw', element_index=[0],
                 data_source=ds, profile_name=["sgen1_p"])

def create_output_writer(net, time_steps, output_dir):
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xlsx", log_variables=list())
    # these variables are saved to the harddisk after / during the time series loop
    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'i_ka')
    return ow


def test_no():
    output_dir = os.path.join(tempfile.gettempdir(), "time_series_example")
    print(output_dir)
    print("Results can be found in your local temp folder: {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 1. create test net
    net = simple_test_net()
    # 2. create (random) data source
    n_timesteps = 100
    profiles, ds = create_data_source(n_timesteps)
    # 3. create controllers (to control P values of the load and the sgen)
    create_controllers(net, ds)

    # time steps to be calculated. Could also be a list with non-consecutive time steps
    time_steps = range(0, n_timesteps)

    # 4. the output writer with the desired results to be stored to files.
    ow = create_output_writer(net, time_steps, output_dir=output_dir)

    # 5. the main time series function
    run_timeseries(net, time_steps)
    print(net.res_line.loading_percent)

    # voltage results
    vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.xlsx")
    vm_pu = pd.read_excel(vm_pu_file, index_col=0)
    vm_pu.plot(label="vm_pu")
    plt.xlabel("time step")
    plt.ylabel("voltage mag. [p.u.]")
    plt.title("Voltage Magnitude")
    plt.grid()
    plt.show()

    # line loading results
    ll_file = os.path.join(output_dir, "res_line", "loading_percent.xlsx")
    line_loading = pd.read_excel(ll_file, index_col=0)
    line_loading.plot(label="line_loading")
    plt.xlabel("time step")
    plt.ylabel("line loading [%]")
    plt.title("Line Loading")
    plt.grid()
    plt.show()

    # load results
    load_file = os.path.join(output_dir, "res_load", "p_mw.xlsx")
    load = pd.read_excel(load_file, index_col=0)
    load.plot(label="load")
    plt.xlabel("time step")
    plt.ylabel("P [MW]")
    plt.title("Load")
    plt.grid()
    plt.show()


def test_discrete_tap():
    output_dir = os.path.join(tempfile.gettempdir(), "time_series_example")
    print(output_dir)
    print("Results can be found in your local temp folder: {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 1. create test net
    net = simple_test_net()

    print(net.bus, net.bus.shape)

    trafo_controller = control.DiscreteTapControl(net=net, tid=0, vm_lower_pu=0.99, vm_upper_pu=1.01)

    print("\n","=========", net.controller, "\n")

    # 2. create (random) data source
    n_timesteps = 100
    profiles, ds = create_data_source(n_timesteps)
    # 3. create controllers (to control P values of the load and the sgen)
    create_controllers(net, ds)

    # time steps to be calculated. Could also be a list with non-consecutive time steps
    time_steps = range(0, n_timesteps)

    # 4. the output writer with the desired results to be stored to files.
    ow = create_output_writer(net, time_steps, output_dir=output_dir)

    # 5. the main time series function
    run_timeseries(net, time_steps,run_control=True,) #progress_function=print
    print(net.res_line.loading_percent)

    # voltage results
    vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.xlsx")
    vm_pu = pd.read_excel(vm_pu_file, index_col=0)
    vm_pu.plot(label="vm_pu")
    plt.xlabel("time step")
    plt.ylabel("voltage mag. [p.u.]")
    plt.title("Voltage Magnitude")
    plt.grid()
    plt.show()

    # line loading results
    ll_file = os.path.join(output_dir, "res_line", "loading_percent.xlsx")
    line_loading = pd.read_excel(ll_file, index_col=0)
    line_loading.plot(label="line_loading")
    plt.xlabel("time step")
    plt.ylabel("line loading [%]")
    plt.title("Line Loading")
    plt.grid()
    plt.show()

    # load results
    load_file = os.path.join(output_dir, "res_load", "p_mw.xlsx")
    load = pd.read_excel(load_file, index_col=0)
    load.plot(label="load")
    plt.xlabel("time step")
    plt.ylabel("P [MW]")
    plt.title("Load")
    plt.grid()
    plt.show()




def test_RL():
    output_dir = os.path.join(tempfile.gettempdir(), "time_series_example")
    print(output_dir)
    print("Results can be found in your local temp folder: {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 1. create test net
    net = simple_test_net()
    trafo_controller = control.DiscreteTapControl(net=net, tid=0, vm_lower_pu=0.99, vm_upper_pu=1.01)

    print("\n","=========", net.controller.shape, "\n")

    # 2. create (random) data source
    n_timesteps = 100
    profiles, ds = create_data_source(n_timesteps)
    # 3. create controllers (to control P values of the load and the sgen)
    create_controllers(net, ds)

    print("\n", "=========", net.controller, "\n")
    # raise NotImplementedError("fdsafas")

    # time steps to be calculated. Could also be a list with non-consecutive time steps
    time_steps = range(0, n_timesteps)

    # 4. the output writer with the desired results to be stored to files.
    ow = create_output_writer(net, time_steps, output_dir=output_dir)

    # 5. the main time series function
    run_timeseries(net, time_steps,run_control=True)
    print(net.res_line.loading_percent)

    # voltage results
    vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.xlsx")
    vm_pu = pd.read_excel(vm_pu_file, index_col=0)
    vm_pu.plot(label="vm_pu")
    plt.xlabel("time step")
    plt.ylabel("voltage mag. [p.u.]")
    plt.title("Voltage Magnitude")
    plt.grid()
    plt.show()

    # line loading results
    ll_file = os.path.join(output_dir, "res_line", "loading_percent.xlsx")
    line_loading = pd.read_excel(ll_file, index_col=0)
    line_loading.plot(label="line_loading")
    plt.xlabel("time step")
    plt.ylabel("line loading [%]")
    plt.title("Line Loading")
    plt.grid()
    plt.show()

    # load results
    load_file = os.path.join(output_dir, "res_load", "p_mw.xlsx")
    load = pd.read_excel(load_file, index_col=0)
    load.plot(label="load")
    plt.xlabel("time step")
    plt.ylabel("P [MW]")
    plt.title("Load")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    test_RL()


