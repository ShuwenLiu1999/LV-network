def R_LV_CIGRE():
    import pandapower as pp
    import pandapower.networks as pn

    # 1) Load original CIGRE LV
    net = pn.create_cigre_network_lv()

    # 2) Find all “R” buses + the “Bus 0” bus
    mask = net.bus.name.str.startswith("Bus R") | (net.bus.name == "Bus 0")
    allowed_buses = set(net.bus.index[mask])

    # 3) Drop any bus not in allowed_buses
    buses_to_drop = set(net.bus.index) - allowed_buses
    net.bus.drop(buses_to_drop, inplace=True)

    # 4) Now for every element table, drop rows that reference a bus outside allowed_buses

    # -- loads & ext_grid --
    net.load.drop(net.load.index[~net.load.bus.isin(allowed_buses)], inplace=True)
    net.ext_grid.drop(net.ext_grid.index[~net.ext_grid.bus.isin(allowed_buses)], inplace=True)

    # -- lines --
    net.line.drop(net.line.index[
                      ~(net.line.from_bus.isin(allowed_buses) & net.line.to_bus.isin(allowed_buses))
                  ], inplace=True)

    # -- transformers --
    net.trafo.drop(net.trafo.index[
                       ~(net.trafo.hv_bus.isin(allowed_buses) & net.trafo.lv_bus.isin(allowed_buses))
                   ], inplace=True)

    # -- switches (bus‐type only) --
    if "switch" in net:
        # for bus‐bus switches, et == "b", element is the other bus
        bus_sw = net.switch[net.switch.et == "b"]
        keep_sw = bus_sw.index[bus_sw.bus.isin(allowed_buses) & bus_sw.element.isin(allowed_buses)]
        net.switch.drop(net.switch.index.difference(keep_sw), inplace=True)

    # -- shunts --
    if "shunt" in net:
        net.shunt.drop(net.shunt.index[~net.shunt.bus.isin(allowed_buses)], inplace=True)

    # -- impedances --
    if "impedance" in net:
        imp = net.impedance
        net.impedance.drop(imp.index[
                               ~(imp.from_bus.isin(allowed_buses) & imp.to_bus.isin(allowed_buses))
                           ], inplace=True)

    # 5) Run PF on the trimmed net
    return net