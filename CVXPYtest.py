import importlib
import math
import numpy as np
import cvxpy as cp
import pandas as pd

GENERATOR_ELEMENT = "Generator"
LOAD_ELEMENT = "Load"
BUS_ELEMENT = "Bus"
LINE_ELEMENT = "Line"
elements_names = [BUS_ELEMENT, LINE_ELEMENT, GENERATOR_ELEMENT, LOAD_ELEMENT]

libs = {}

def initialize_network(network_dict):
    nper = network_dict["stats"]["nper"] # Number of periods
    nbuses = len(network_dict["elements"][BUS_ELEMENT]) # Number of buses
    nlines = len(network_dict["elements"][LINE_ELEMENT]) # Number of lines
    nvar = 0
    network_dict["cost"] = {}

    # Import elements libraries
    for element in network_dict["elements"]:
        libs[element] = importlib.import_module(f"elements.{element.lower()}")

    # Initialize buses
    for idx, value in network_dict["elements"][BUS_ELEMENT].items():
        libs[BUS_ELEMENT].initialize(network_dict, idx)
        if value["attrs"]["slack"]:
            network_dict["stats"]["slack_bus"] = idx

    # Initialize lines
    for idx, value in network_dict["elements"][LINE_ELEMENT].items():
        libs[LINE_ELEMENT].initialize(network_dict, idx)

    # Initialize the rest of the elements
    for element in network_dict["elements"]:
        for idx, value in network_dict["elements"][element].items():
            if element not in [BUS_ELEMENT, LINE_ELEMENT]:
                libs[element].initialize(network_dict, idx)
                for attr_key, attr_value in value["vars"].items():
                    attr_value["id"] = nvar
                    bus_index = value["attrs"]["bus"]
                    if "p_pu" in attr_key:
                        network_dict["elements"][BUS_ELEMENT][bus_index]["p_dev_idxs"].append(nvar)
                    elif "q_pu" in attr_key:
                        network_dict["elements"][BUS_ELEMENT][bus_index]["q_dev_idxs"].append(nvar)
                    nvar += 1
    network_dict["stats"]["nvar"] = nvar
    network_dict["var_values"] = np.zeros((nvar, nper))
    network_dict["cost"]["c1"] = np.zeros((nvar, nper))
    network_dict["cost"]["c2"] = np.zeros((nvar, nper))
    network_dict["vm_min"] = np.zeros((nbuses, nper))
    network_dict["vm_max"] = np.zeros((nbuses, nper))
    network_dict["var_min"] = np.zeros((nvar, nper))
    network_dict["var_max"] = np.zeros((nvar, nper))
    network_dict["lines_s_max"] = np.zeros((nlines))
    network_dict["bus_base_p_load"] = np.zeros((nbuses, nper))
    network_dict["bus_base_q_load"] = np.zeros((nbuses, nper))

def load_scenario(scenario):
    params_file = pd.ExcelFile(scenario)

    network_dict = {}
    scenario_config = params_file.parse("config")
    network_dict["stats"] = {}
    for key, value in scenario_config.items():
        network_dict["stats"][key] = int(value[0])

    network_dict["elements"] = {}
    # Initialize network_dict with the data from the params file into 'attrs' key
    for sheet_name in params_file.sheet_names:
        if sheet_name in elements_names:
            _params = params_file.parse(sheet_name, index_col=0).to_dict('index')
            if _params:
                network_dict["elements"][sheet_name] = {}
            for idx, value in _params.items():  # idx is the index, value is the dict of attributes
                network_dict["elements"][sheet_name][idx] = {}
                network_dict["elements"][sheet_name][idx]['attrs'] = {}
                network_dict["elements"][sheet_name][idx]['attrs']['index'] = idx
                for attr, value2 in value.items():  # attr is the attribute, value2 is the value
                    if value2 == "Inf":
                        network_dict["elements"][sheet_name][idx]['attrs'][attr] = math.inf
                    elif value2 == "-Inf":
                        network_dict["elements"][sheet_name][idx]['attrs'][attr] = -math.inf
                    else:
                        network_dict["elements"][sheet_name][idx]['attrs'][attr] = value2

    print(f"Initializing scenario {scenario}...", end=" ")
    initialize_network(network_dict)
    update_network(network_dict)
    print("Done")
    return network_dict

def build_socp_model(network_dict):
    print("Building SOCP model...", end=" ")
    nvar = network_dict["stats"]["nvar"]
    nper = network_dict["stats"]["nper"]
    Bus = len(network_dict["elements"]["Bus"])
    Line = len(network_dict["elements"]["Line"])
    bus_p_load = cp.Parameter((Bus, nper), name="bus_p_load")
    bus_q_load = cp.Parameter((Bus, nper), name="bus_q_load")
    p = cp.Variable((Bus, nper), name="p")
    q = cp.Variable((Bus, nper), name="q")
    p_f = cp.Variable((Line, nper), name="p_f")
    q_f = cp.Variable((Line, nper), name="q_f")
    p_t = cp.Variable((Line, nper), name="p_t")
    q_t = cp.Variable((Line, nper), name="q_t")
    dev_variables = cp.Variable((nvar, nper), name="dev_variables") # devices variables (desicion variables)
    dev_min = cp.Parameter((nvar, nper), name="dev_min")
    dev_max = cp.Parameter((nvar, nper), name="dev_max")
    s_pu_max = cp.Parameter((Line,), name="s_pu_max")
    cost1 = cp.Parameter((nvar, nper), name="cost1")
    cost2_sqrt = cp.Parameter((nvar, nper), name="cost2_sqrt")
    # SOCP variables
    c_l = cp.Variable((Line, nper), name="c_l")  # corresponds to lines c's
    c_b = cp.Variable((Bus, nper), name="c_b")  # corresponds to buses c's
    s = cp.Variable((Line, nper), name="s")

    constraints = []

    # bounds constraints
    constraints += [dev_variables >= dev_min, dev_variables <= dev_max]

    # bus devices-inyection balance constraints
    bus_dict = network_dict["elements"][BUS_ELEMENT]
    p_dev_sum = cp.vstack(
        [cp.sum(dev_variables[bus_dict[i]["p_dev_idxs"], :], axis=0) for i in range(Bus)])
    q_dev_sum = cp.vstack(
        [cp.sum(dev_variables[bus_dict[i]["q_dev_idxs"], :], axis=0) for i in range(Bus)])
    constraints += [p == bus_p_load + p_dev_sum,
                    q == bus_q_load + q_dev_sum]
    
    # relaxed power flow constraints
    pbus = np.zeros((Bus, nper)).tolist()
    qbus = np.zeros((Bus, nper)).tolist()
    G_l = network_dict["Y"]["G_lines"]
    B_l = network_dict["Y"]["B_lines"]
    G_SH = network_dict["Y"]["G_SH"]
    B_SH = network_dict["Y"]["B_SH"]
    vm_min_sq = network_dict["vm_min"] ** 2
    vm_max_sq = network_dict["vm_max"] ** 2
        
    constraints += [p_f == cp.multiply(c_b[network_dict["from_buses"], :], G_SH[:, None] - G_l[:, None]) + cp.multiply(c_l, G_l[:, None]) + cp.multiply(s, B_l[:, None])]
    constraints += [q_f == -cp.multiply(c_b[network_dict["from_buses"], :], B_SH[:, None] - B_l[:, None]) - cp.multiply(c_l, B_l[:, None]) + cp.multiply(s, G_l[:, None])]
    constraints += [p_t == cp.multiply(c_b[network_dict["to_buses"], :], G_SH[:, None] - G_l[:, None]) + cp.multiply(c_l, G_l[:, None]) - cp.multiply(s, B_l[:, None])]
    constraints += [q_t == -cp.multiply(c_b[network_dict["to_buses"], :], B_SH[:, None] - B_l[:, None]) - cp.multiply(c_l, B_l[:, None]) - cp.multiply(s, G_l[:, None])]

    #bus lines-inyection balance constraints
    for i in range(Bus):
        pbus[i] = cp.sum(p_f[bus_dict[i]['from_lines'], :], axis=0) + cp.sum(p_t[bus_dict[i]['to_lines'], :], axis=0)
        qbus[i] = cp.sum(q_f[bus_dict[i]['from_lines'], :], axis=0) + cp.sum(q_t[bus_dict[i]['to_lines'], :], axis=0)
    pbus[i] = cp.hstack(pbus[i])
    qbus[i] = cp.hstack(qbus[i])

    constraints += [p == cp.vstack(pbus)]
    constraints += [q == cp.vstack(qbus)]

    # socp constraints
    M1 = (cp.vec(c_b[network_dict["from_buses"], :]) - cp.vec(c_b[network_dict["to_buses"], :])) / 2
    M2 = (cp.vec(c_b[network_dict["from_buses"], :]) + cp.vec(c_b[network_dict["to_buses"], :])) / 2
    M3 = cp.hstack([M1, M1])
    M4 = cp.hstack([M2, M2])
    M5 = cp.hstack([cp.vec(c_l), cp.vec(c_l)])
    M6 = cp.hstack([cp.vec(s), -cp.vec(s)])
    M7 = cp.vstack([M3, M5, M6])
    constraints += [cp.SOC(M4, M7)]

    # voltage bounds constraints
    constraints += [c_b <= vm_max_sq]
    constraints += [c_b >= vm_min_sq]

    # max power flow constraints
    constraints += [p_f <= s_pu_max[:, None],
                    p_t <= s_pu_max[:, None]]

    # Objective
    sn_mva = network_dict["stats"]["sn_mva"]
    _dev_variables = cp.vec(dev_variables)
    _cost1 = cp.vec(cost1)
    _cost2_sqrt = cp.diag(cp.vec(cost2_sqrt))
    quad_form = cp.sum_squares(_cost2_sqrt @ _dev_variables)
    objective = _cost1.T @ _dev_variables + sn_mva * quad_form

    # Problem
    model = cp.Problem(cp.Minimize(objective), constraints)

    # Parameters initialization
    bus_p_load.value = network_dict['bus_base_p_load']
    bus_q_load.value = network_dict['bus_base_q_load']
    dev_min.value = network_dict["var_min"]
    dev_max.value = network_dict["var_max"]
    cost1.value = network_dict["cost"]["c1"]
    cost2_sqrt.value = np.sqrt(network_dict["cost"]["c2"])
    s_pu_max.value = network_dict["lines_s_max"]

    print("Done")
    return model

def update_admittance_matrix(network_dict):
    """
    Updates the admittance matrix for the network: Y = {"G": G, "B": B, "G_SH": G_SH, "B_SH": B_SH}
    """
    nbuses = len(network_dict["elements"][BUS_ELEMENT])
    nlines = len(network_dict["elements"][LINE_ELEMENT])
    G_l = np.zeros(nlines)
    B_l = np.zeros(nlines)
    G_SH = np.zeros(nlines)
    B_SH = np.zeros(nlines)
    G_b = np.zeros(nbuses)
    B_b = np.zeros(nbuses)
    for index, values in network_dict["elements"][LINE_ELEMENT].items():
        if values["attrs"]["is_active"]:
            i = values["attrs"]["from_bus"]
            j = values["attrs"]["to_bus"]
            b_sh = values["params"]["b_sh_pu"]
            # g_sh = values["attrs"]["g_sh_pu"]
            g_sh = 0
            g = values["params"]["g_pu"]
            b = values["params"]["b_pu"]
            G_l[index] = -g
            B_l[index] = -b
            G_b[i] += g + g_sh / 2
            G_b[j] += g + g_sh / 2
            B_b[i] += b + b_sh / 2
            B_b[j] += b + b_sh / 2
            B_SH[index] = b_sh / 2
            G_SH[index] = g_sh / 2

    network_dict["Y"] = {"G_lines": G_l, "B_lines": B_l, "G_buses": G_b, "B_buses": B_b, "G_SH": G_SH, "B_SH": B_SH}


def update_bus_base_load(network_dict):
    """
    Returns a list with the base load of each bus
    """
    nper = network_dict["stats"]["nper"]
    nbuses = len(network_dict["elements"]["Bus"])
    network_dict["bus_base_p_load"] = np.zeros((nbuses, nper))
    network_dict["bus_base_q_load"] = np.zeros((nbuses, nper))
    for index, values in network_dict["elements"]["Load"].items():
        if values["attrs"]["is_active"]:
            i = values["attrs"]["bus"]
            network_dict["bus_base_p_load"][i] += values["params"]["p_pu_base"]
            network_dict["bus_base_q_load"][i] += values["params"]["q_pu_base"]


def update_connection_matrix(network_dict):
    network_dict["from_buses"] = []
    network_dict["to_buses"] = []
    for index, values in network_dict["elements"][LINE_ELEMENT].items():
        if values["attrs"]["is_active"]:
            i = values["attrs"]["from_bus"]
            j = values["attrs"]["to_bus"]
            network_dict["from_buses"].append(i)
            network_dict["to_buses"].append(j)


def update_cost(network_dict):
    for key, idx_values in network_dict["elements"].items():
        for idx, value in idx_values.items():
            VARS = libs[key].VARS
            C1 = libs[key].C1
            C2 = libs[key].C2
            for i, var in enumerate(VARS):
                id = value["vars"][var]["id"]
                if C1[i]:
                    network_dict["cost"]["c1"][id] = value["params"][C1[i]]
                if C2[i]:
                    network_dict["cost"]["c2"][id] = value["params"][C2[i]]


def update_elements_parameters(network_dict):
    for key, idx_values in network_dict["elements"].items():
        for idx, values in idx_values.items():  # idx is the index, values is the dict of attrs
            libs[key].update_parameters(network_dict, idx)


def update_lines_capacities(network_dict):
    """
    Returns a dictionary with the capacity of the lines
    """
    sn_mva = network_dict["stats"]["sn_mva"]
    for index, values in network_dict["elements"]["Line"].items():
        if values["attrs"]["is_active"]:
            network_dict["lines_s_max"][index] = values["attrs"]["s_mva_max"] / sn_mva


def update_network(network_dict):
    update_elements_parameters(network_dict)
    update_admittance_matrix(network_dict)
    update_lines_capacities(network_dict)
    update_connection_matrix(network_dict)
    update_bus_base_load(network_dict)
    update_variables_bounds(network_dict)
    update_voltage_bounds(network_dict)
    update_cost(network_dict)


def update_variables_bounds(network_dict):
    """
    For each variable in networkdict, updates the parameters dev_min and dev_max with the
    bounds. For this, it searches for the variable in network_dict and updates its bounds from the attribute
    called by adding "_min" and "_max" to the variable name.
    """
    for key, idx_values in network_dict["elements"].items():
        for index, value in idx_values.items():
            for attr_key, attr_value in value["vars"].items():
                if "p_pu" in attr_key or "q_pu" in attr_key:
                    id = attr_value["id"]
                    network_dict["var_min"][id] = value["params"][f"{attr_key}_min"]
                    network_dict["var_max"][id] = value["params"][f"{attr_key}_max"]


def update_voltage_bounds(network_dict):
    """
    Returns a dictionary with the voltage bounds for the network
    """
    for idx, idx_values in network_dict["elements"][BUS_ELEMENT].items():
        network_dict["vm_min"][idx] = idx_values["attrs"]["v_pu_min"]
        network_dict["vm_max"][idx] = idx_values["attrs"]["v_pu_max"]


if __name__ == '__main__':
    network_dict = load_scenario("118buses.xlsx")
    mainop = build_socp_model(network_dict)
    mainop.solve(solver=cp.ECOS , verbose=True, max_iters=1000000)
    print(network_dict['stats'])
