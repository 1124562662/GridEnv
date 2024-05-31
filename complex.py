#import the pandapower module
import pandapower as pp
import pandas as pd

if __name__ == "__main__":
    #create an empty network
    net = pp.create_empty_network()

    # Double busbar
    pp.create_bus(net, name='Double Busbar 1', vn_kv=380, type='b')
    pp.create_bus(net, name='Double Busbar 2', vn_kv=380, type='b')
    for i in range(10):
        pp.create_bus(net, name='Bus DB T%s' % i, vn_kv=380, type='n')
    for i in range(1, 5):
        pp.create_bus(net, name='Bus DB %s' % i, vn_kv=380, type='n')

    # Single busbar
    pp.create_bus(net, name='Single Busbar', vn_kv=110, type='b')
    for i in range(1, 6):
        pp.create_bus(net, name='Bus SB %s' % i, vn_kv=110, type='n')
    for i in range(1, 6):
        for j in [1, 2]:
            pp.create_bus(net, name='Bus SB T%s.%s' % (i, j), vn_kv=110, type='n')

    # Remaining buses
    for i in range(1, 5):
        pp.create_bus(net, name='Bus HV%s' % i, vn_kv=110, type='n')

    # show bustable
    print(net.bus)

    hv_lines = pd.read_csv('example_advanced/hv_lines.csv', sep=';', header=0, decimal=',')

    # create lines
    for _, hv_line in hv_lines.iterrows():
        from_bus = pp.get_element_index(net, "bus", hv_line.from_bus)
        to_bus = pp.get_element_index(net, "bus", hv_line.to_bus)
        pp.create_line(net, from_bus, to_bus, length_km=hv_line.length, std_type=hv_line.std_type,
                       name=hv_line.line_name, parallel=hv_line.parallel)

    hv_bus = pp.get_element_index(net, "bus", "Bus DB 2")
    lv_bus = pp.get_element_index(net, "bus", "Bus SB 1")
    pp.create_transformer_from_parameters(net, hv_bus, lv_bus, sn_mva=300, vn_hv_kv=380, vn_lv_kv=110, vkr_percent=0.06,
                                          vk_percent=8, pfe_kw=0, i0_percent=0, tp_pos=0, shift_degree=0,
                                          name='EHV-HV-Trafo')

    hv_bus_sw = pd.read_csv('example_advanced/hv_bus_sw.csv', sep=';', header=0, decimal=',')
    # Bus-bus switches
    for _, switch in hv_bus_sw.iterrows():
        from_bus = pp.get_element_index(net, "bus", switch.from_bus)
        to_bus = pp.get_element_index(net, "bus", switch.to_bus)
        pp.create_switch(net, from_bus, to_bus, et=switch.et, closed=switch.closed, type=switch.type,
                         name=switch.bus_name)

    # Bus-line switches
    hv_buses = net.bus[(net.bus.vn_kv == 380) | (net.bus.vn_kv == 110)].index
    hv_ls = net.line[(net.line.from_bus.isin(hv_buses)) & (net.line.to_bus.isin(hv_buses))]
    for _, line in hv_ls.iterrows():
        pp.create_switch(net, line.from_bus, line.name, et='l', closed=True, type='LBS',
                         name='Switch %s - %s' % (net.bus.name.at[line.from_bus], line['name']))
        pp.create_switch(net, line.to_bus, line.name, et='l', closed=True, type='LBS',
                         name='Switch %s - %s' % (net.bus.name.at[line.to_bus], line['name']))

    # Trafo-line switches
    pp.create_switch(net, pp.get_element_index(net, "bus", 'Bus DB 2'),
                     pp.get_element_index(net, "trafo", 'EHV-HV-Trafo'), et='t', closed=True, type='LBS',
                     name='Switch DB2 - EHV-HV-Trafo')
    pp.create_switch(net, pp.get_element_index(net, "bus", 'Bus SB 1'),
                     pp.get_element_index(net, "trafo", 'EHV-HV-Trafo'), et='t', closed=True, type='LBS',
                     name='Switch SB1 - EHV-HV-Trafo')

    pp.create_ext_grid(net, pp.get_element_index(net, "bus", 'Double Busbar 1'), vm_pu=1.03, va_degree=0,
                       name='External grid',
                       s_sc_max_mva=10000, rx_max=0.1, rx_min=0.1)

    hv_loads = pd.read_csv('example_advanced/hv_loads.csv', sep=';', header=0, decimal=',')

    for _, load in hv_loads.iterrows():
        bus_idx = pp.get_element_index(net, "bus", load.bus)
        pp.create_load(net, bus_idx, p_mw=load.p, q_mvar=load.q, name=load.load_name)

    pp.create_gen(net, pp.get_element_index(net, "bus", 'Bus HV4'), vm_pu=1.03, p_mw=100, name='Gas turbine')

    pp.create_sgen(net, pp.get_element_index(net, "bus", 'Bus SB 5'), p_mw=20, q_mvar=4, sn_mva=45,
                   type='WP', name='Wind Park')