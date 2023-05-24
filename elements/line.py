import numpy as np

VARS = []
C1 = []
C2 = []

def initialize(network_dict, i):
    nper = network_dict['stats']['nper']
    line_dict = network_dict['elements']["Line"][i]

    line_dict['params'] = {}
    line_dict['params']['g_pu'] = None
    line_dict['params']['b_pu'] = None
    line_dict['params']['s_pu_max'] = None
    line_dict['params']['is_active'] = None
    line_dict["attrs"]["g_sh_pu"] = 0

    line_dict["vars"] = {}
    for var in VARS:
        line_dict["vars"][var] = {'id': i, 'value': np.zeros(nper)}

def update_parameters(network_dict, i):
    line_dict = network_dict['elements']["Line"][i]
    sn_mva = network_dict['stats']['sn_mva']
    v_kv_from = network_dict['elements']["Bus"][line_dict['attrs']['from_bus']]['attrs']['vn_kv']

    zb = (v_kv_from ** 2) / sn_mva

    r_pu = line_dict["attrs"]["r"] / zb
    x_pu = line_dict["attrs"]["x"] / zb
    line_dict['params']['b_sh_pu'] = line_dict["attrs"]["b_sh"] / zb
    line_dict['params']['g_pu'] = (1 / complex(r_pu, x_pu)).real
    line_dict['params']['b_pu'] = (1 / complex(r_pu, x_pu)).imag
    line_dict['params']['s_pu_max'] = line_dict['attrs']['s_mva_max'] / sn_mva