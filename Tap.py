import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.control
import json


def add_trafo_connection(net, hv_bus, trafotype="2W"):
    cb = pp.create_bus(net, vn_kv=0.4)
    pp.create_load(net, cb, 0.2, 0.05)

    if trafotype == "3W":
        cbm = pp.create_bus(net, vn_kv=0.9)
        pp.create_load(net, cbm, 0.1, 0.03)
        pp.create_transformer3w_from_parameters(net, hv_bus=hv_bus, mv_bus=cbm, lv_bus=cb,
                                                vn_hv_kv=20., vn_mv_kv=0.9, vn_lv_kv=0.45, sn_hv_mva=0.6,
                                                sn_mv_mva=0.5, sn_lv_mva=0.4, vk_hv_percent=1.,
                                                vk_mv_percent=1., vk_lv_percent=1., vkr_hv_percent=0.3,
                                                vkr_mv_percent=0.3, vkr_lv_percent=0.3, pfe_kw=0.2,
                                                i0_percent=0.3, tap_neutral=0., tap_pos=2,
                                                tap_step_percent=1., tap_min=-2, tap_max=2)
    else:
        pp.create_transformer(net, hv_bus=hv_bus, lv_bus=cb, std_type="0.25 MVA 20/0.4 kV", tap_pos=2)


def create_net():
    net = pp.create_empty_network()
    vn_kv = 20
    b1 = pp.create_bus(net, vn_kv=vn_kv)
    pp.create_ext_grid(net, b1, vm_pu=1.01)
    b2 = pp.create_bus(net, vn_kv=vn_kv)
    l1 = pp.create_line_from_parameters(net, b1, b2, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                        c_nf_per_km=300, max_i_ka=.2, df=.8)
    for i in range(2):
        add_trafo_connection(net, b2)

    return net


if __name__ == "__main__":
    net = create_net()
    pp.control.create_trafo_characteristics(net, 'trafo', 0, 'vk_percent',
                                            [-2, -1, 0, 1, 2], [5, 5.2, 6, 6.8, 7])  # single mode
    pp.control.create_trafo_characteristics(net, 'trafo', [0], 'vkr_percent',
                                            [[-2, -1, 0, 1, 2]], [[1.3, 1.4, 1.44, 1.5, 1.6]])  # multiple indices

    pp.control.plot_characteristic(net.characteristic.object.at[0], -2, 2,
                                   xlabel='Tap position "tap_pos"', ylabel='Value of "vk_percent"')

    pp.control.plot_characteristic(net.characteristic.object.at[1], -2, 2,
                                   xlabel='Tap position "tap_pos"', ylabel='Value of "vkr_percent"')

    pp.runpp(net)