import numpy as np

VARS = ['p_pu', 'q_pu']
C1 = ['c1', 0]
C2 = ['c2', 0]


def initialize(network_dict, i):
    nper = network_dict['stats']['nper']
    gen_dict = network_dict['elements']["Generator"][i]

    gen_dict['params'] = {}
    gen_dict['params']['p_pu_ini'] = None
    gen_dict['params']['p_pu_min'] = None
    gen_dict['params']['p_pu_max'] = None
    gen_dict['params']['q_pu_min'] = None
    gen_dict['params']['q_pu_max'] = None
    gen_dict['params']['d_pu_max'] = None
    gen_dict['params']['c1'] = None
    gen_dict['params']['c2'] = None

    gen_dict["vars"] = {}
    for var in VARS:
        gen_dict["vars"][var] = {'id': 0, 'value': np.zeros(nper)}


def update_parameters(network_dict, i):
    sn_mva = network_dict['stats']['sn_mva']
    nper = network_dict['stats']['nper']
    pdur = network_dict['stats']['pdur']
    gen_dict = network_dict['elements']["Generator"][i]

    gen_dict['params']['p_pu_ini'] = gen_dict['attrs']['p_mw_ini'] / sn_mva
    gen_dict['params']['d_pu_max'] = gen_dict['attrs']['d_mw_max'] / sn_mva  # maximum ramp in one hour
    gen_dict['params']['p_pu_min'] = np.full(nper, gen_dict['attrs']['p_mw_min']) / sn_mva
    gen_dict['params']['q_pu_min'] = np.full(nper, gen_dict['attrs']['q_mvar_min']) / sn_mva
    gen_dict['params']['p_pu_max'] = np.full(nper, gen_dict['attrs']['p_mw_max']) / sn_mva
    gen_dict['params']['q_pu_max'] = np.full(nper, gen_dict['attrs']['q_mvar_max']) / sn_mva
    gen_dict['params']['p_pu_min'][0] = max(
        gen_dict['params']['p_pu_ini'] - gen_dict['params']['d_pu_max'] * pdur / 60,
        gen_dict['params']['p_pu_min'][0])
    gen_dict['params']['p_pu_max'][0] = min(
        gen_dict['params']['p_pu_ini'] + gen_dict['params']['d_pu_max'] * pdur / 60,
        gen_dict['params']['p_pu_max'][0])

    gen_dict['params']['c1'] = np.full(nper, gen_dict['attrs']['c1'])
    gen_dict['params']['c2'] = np.full(nper, gen_dict['attrs']['c2'])

