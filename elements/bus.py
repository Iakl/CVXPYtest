import numpy as np


VARS = ['v_pu', 'ang_rad']
C1 = [0, 0]
C2 = [0, 0]

def initialize(network_dict, i):
    nper = network_dict['stats']['nper']
    bus_dict = network_dict['elements']["Bus"][i]

    bus_dict["p_dev_idxs"] = []
    bus_dict["q_dev_idxs"] = []
    bus_dict["lines"] = []
    bus_dict["from_lines"] = []
    bus_dict["to_lines"] = []
    bus_dict['params'] = {}
    bus_dict['params']['v_pu_min'] = None
    bus_dict['params']['v_pu_max'] = None
    bus_dict['params']['v_pu_set'] = None
    bus_dict['params']['bs_pu'] = None
    bus_dict['params']['gs_pu'] = None
    bus_dict['params']['b_pu'] = 0
    bus_dict['params']['g_pu'] = 0

    bus_dict["vars"] = {}
    for var in VARS:
        bus_dict["vars"][var] = {'id': i, 'value': np.zeros(nper)}


def update_parameters(network_dict, i):
    sn_mva = network_dict['stats']['sn_mva']
    bus_dict = network_dict['elements']["Bus"][i]

    bus_dict['params']['bs_pu'] = bus_dict['attrs']['bs_mvar'] / sn_mva
    bus_dict['params']['gs_pu'] = bus_dict['attrs']['gs_mw'] / sn_mva

    for line, value in network_dict['elements']["Line"].items():
        if value['attrs']['from_bus'] == i:
            bus_dict['lines'].append(line)
            bus_dict['from_lines'].append(line)
        elif value['attrs']['to_bus'] == i:
            bus_dict['lines'].append(line)
            bus_dict['to_lines'].append(line)
