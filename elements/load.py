import numpy as np

VARS = ['p_pu_diff_pos', 'p_pu_diff_neg', 'q_pu_diff_pos', 'q_pu_diff_neg']
C1 = ['flex_price_up', 'flex_price_down', 0, 0]
C2 = [0, 0, 0, 0]


def initialize(network_dict, i):
    nper = network_dict['stats']['nper']
    load_dict = network_dict['elements']["Load"][i]

    load_dict['params'] = {}
    load_dict['params']['q_pu_base'] = None
    load_dict['params']['p_pu_base'] = None
    load_dict['params']['flex_price_up'] = None
    load_dict['params']['flex_price_down'] = None
    reset_bounds(load_dict['params'], nper)

    load_dict["vars"] = {}
    for var in VARS:
        load_dict["vars"][var] = {'id': 0, 'value': np.zeros(nper)}


def reset_bounds(params_dict, nper):
    params_dict['p_pu_diff_pos_min'] = np.zeros(nper)
    params_dict['p_pu_diff_pos_max'] = np.zeros(nper)
    params_dict['q_pu_diff_pos_min'] = np.zeros(nper)
    params_dict['q_pu_diff_pos_max'] = np.zeros(nper)
    params_dict['p_pu_diff_neg_min'] = np.zeros(nper)
    params_dict['p_pu_diff_neg_max'] = np.zeros(nper)
    params_dict['q_pu_diff_neg_min'] = np.zeros(nper)
    params_dict['q_pu_diff_neg_max'] = np.zeros(nper)


def update_parameters(network_dict, i):
    sn_mva = network_dict['stats']['sn_mva']
    nper = network_dict['stats']['nper']
    load_dict = network_dict['elements']["Load"][i]

    load_dict['params']['p_pu_base'] = np.full(nper, load_dict['attrs']['p_mw']) / sn_mva
    load_dict['params']['q_pu_base'] = np.full(nper, load_dict['attrs']['q_mvar']) / sn_mva

    load_dict['params']['flex_price_up'] = np.full(nper, load_dict['attrs']['flex_price_up'])
    load_dict['params']['flex_price_down'] = np.full(nper, load_dict['attrs']['flex_price_down'])

