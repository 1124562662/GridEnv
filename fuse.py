import pandapower as pp


def fuse_example_net1():
    net = pp.create_empty_network()
    # create buses
    pp.create_buses(net, nr_buses=4, vn_kv=[20, 0.4, 0.4, 0.4, 0.4], index=[0, 1, 2, 3, 4], name=None, type="n",
                    geodata=[(0, 0), (0, -2), (0, -4), (0, -6), (0, -8)])

    # create external grid
    pp.create_ext_grid(net, 0, vm_pu=1.0, va_degree=0, s_sc_max_mva=100, s_sc_min_mva=50, rx_max=0.1, rx_min=0.1)
    pp.create_lines_from_parameters(net, from_buses=[1, 2], to_buses=[2, 3], length_km=[0.1, 0.1], r_ohm_per_km=0.2067,
                                    x_ohm_per_km=0.080424, c_nf_per_km=261, name=None, index=[0, 1], max_i_ka=0.27)

    net.line["endtemp_degree"] = 250
    # create transformer
    pp.create_transformer(net, hv_bus=0, lv_bus=1, std_type="0.63 MVA 20/0.4 kV")

    # Define trafo fuses
    pp.create_switches(net, buses=[0, 1], elements=[0, 0], et='t', type="fuse")

    # Define line fuses
    pp.create_switches(net, buses=[1, 2], elements=[0, 1], et='l', type="fuse")

    # Define load fuse (bus-bus switch)
    pp.create_switch(net, bus=3, element=4, et='b', type="fuse", z_ohm=0.0001)

    # define load
    pp.create_load(net, bus=4, p_mw=0.1, q_mvar=0, const_z_percent=0, const_i_percent=0, sn_mva=.1,
                   name=None, scaling=1., index=0)
    return net

if __name__ == "__main__":
    net = fuse_example_net1()
    # diagnoses the faulty network
    pp.diagnostic(net)