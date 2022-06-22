from platform import node
from select import select
import numpy as np
import pandas as pd
from pulp import *
import pickle
from cvrptw import read_input_cvrptw
from tsp import get_tsp_solution


def path_selection(num_customers, 
                   paths_dict,
                   paths_cost_dict,
                   paths_customers_dict,
                   number_of_paths=None,
                   lp_file_name=None,
                   binary_model=False,
                   mip_gap=0.001,
                   solver_time_limit_minutes=10,
                   enable_solution_messaging=1,
                   solver_type='PULP_CBC_CMD'
                   ):
    customers_var = [f"Customer_{i}" for i in range(1, num_customers+1)]
    master_model = LpProblem("MA_CVRPTW", LpMinimize)
    if binary_model:
        path_var = LpVariable.dicts("Path", paths_dict.keys(), 0, 1, LpBinary)
    else:
        path_var = LpVariable.dicts("Path", paths_dict.keys(), 0, 1, LpContinuous)
    print('Master model objective function')
    master_model += lpSum(paths_cost_dict[path] * path_var[path] for path in paths_dict.keys())

    print('Each customer belongs to one path')
    for customer in customers_var:
        master_model += lpSum(
            [paths_customers_dict[path, customer] * path_var[path] for path in
                paths_dict.keys()]) == 1, "Customer" + str(customer)

    if number_of_paths is not None:
        master_model += lpSum(
            [path_var[path] for path in
                paths_dict.keys()]) <= number_of_paths, "No of Vehicles"

    if lp_file_name is not None:
        master_model.writeLP('{}.lp'.format(str(lp_file_name)))

    if solver_type == 'PULP_CBC_CMD':
        master_model.solve(PULP_CBC_CMD(
            msg=enable_solution_messaging,
            timeLimit=60*solver_time_limit_minutes,
            gapRel=mip_gap)
        )
    elif solver_type == "GUROBI_CMD":
        solver = getSolver('GUROBI_CMD', msg=enable_solution_messaging,
            timeLimit=60*solver_time_limit_minutes)
        master_model.solve(solver)

    print('Master Model Status = {}'.format(LpStatus[master_model.status]))
    if master_model.status == 1:
        solution_master_model_objective = value(master_model.objective)
        total_cost = 0.0
        print('Master model objective = {}'.format(str(solution_master_model_objective)))
        price = {}
        for customer in customers_var:
            if solver_type == "GUROBI_CMD":
                price[customer] = float(master_model.constraints["Customer" + str(customer)].pi)
            else:
                price[customer] = float(master_model.constraints["Customer" + str(customer)].pi)
        solution_master_path = []
        for path in path_var.keys():
            if path_var[path].value() and path_var[path].value() > 0:
                solution_master_path.append({'PATH_NAME': path,
                                             'VALUE': path_var[path].value(),
                                             'PATH': paths_dict[path]
                                             })
                total_cost += paths_cost_dict[path]
                print(f"path_name: {path}, {paths_dict[path]}")
        solution_master_path = pd.DataFrame(solution_master_path)
        solution_master_path['OBJECTIVE'] = solution_master_model_objective
        return total_cost, price
    else:
        raise Exception('No Solution Exists')

def cvrptw_one_vehicle(selected_customers, 
                       truck_capacity, distance_matrix, 
                       distance_warehouses, demands, service_time,
                       earliest_start, latest_end, 
                       max_horizon,
                       prices = None,
                       lp_file_name = None,
                       bigm=1000000,
                       mip_gap=0.001,
                       solver_time_limit_minutes=10,
                       enable_solution_messaging=1,
                       solver_type='PULP_CBC_CMD'):
    
    depot = "Customer_0"
    num_customers = len(selected_customers) + 1
    local_customers_var = [f"Customer_{i}" for i in range(num_customers)]
    local_transit_cost = np.zeros((num_customers, num_customers))
    for i in range(num_customers):
        for j in range(num_customers):
            if i == 0 and j == 0: continue
            elif i == 0: local_transit_cost[i, j] = distance_warehouses[selected_customers[j-1]]
            elif j == 0: local_transit_cost[i, j] = distance_warehouses[selected_customers[i-1]]
            else: local_transit_cost[i, j] = distance_matrix[selected_customers[i-1]][selected_customers[j-1]]

    local_assignment_var_dict = {}
    local_transit_cost_dict = {}
    for i in range(num_customers):
        for j in range(num_customers):
            local_assignment_var_dict[f"Customer_{i}", f"Customer_{j}"] = 0
            local_transit_cost_dict[f"Customer_{i}", f"Customer_{j}"] = local_transit_cost[i, j]

    local_service_time = {}
    local_demands = {}
    local_earliest_start = {}
    local_latest_end = {}
    for i in range(num_customers):
        key = local_customers_var[i]
        local_demands[key] = (0 if i == 0 else demands[selected_customers[i-1]])
        local_earliest_start[key] = (0 if i == 0 else earliest_start[selected_customers[i-1]])
        local_latest_end[key] = (max_horizon if i == 0 else latest_end[selected_customers[i-1]])
        local_service_time[key] = (0 if i == 0 else service_time[selected_customers[i-1]])

    # sub problem
    sub_model = LpProblem("SU_CVRPTW", LpMinimize)
    time_var = LpVariable.dicts("Time", local_customers_var, 0, None, LpContinuous)
    assignment_var = LpVariable.dicts("Assign", local_assignment_var_dict.keys(), 0, 1, LpBinary)

    print('objective function')
    max_transportation_cost = np.max(list(local_transit_cost_dict.values()))
    
    if prices is None:
        sub_model += lpSum(
            (local_transit_cost_dict[from_loc, to_loc]-max_transportation_cost) * assignment_var[from_loc, to_loc]
            for from_loc, to_loc in local_assignment_var_dict.keys())
    else:
        prices[depot] = 0.0
        sub_model += lpSum(
            (local_transit_cost_dict[from_loc, to_loc]-max_transportation_cost-prices[from_loc]) * assignment_var[from_loc, to_loc]
            for from_loc, to_loc in local_assignment_var_dict.keys())
    # Each vehicle should leave from a depot
    print('Each vehicle should leave from a depot')
    sub_model += lpSum([assignment_var[depot, customer]
                                for customer in local_customers_var]) == 1, "entryDepotConnection"

    # Flow in Flow Out
    print('Flow in Flow out')
    for customer in local_customers_var:
        sub_model += (assignment_var[customer, customer] == 0.0, f"no loop for {customer}")
        sub_model += lpSum(
            [assignment_var[from_loc, customer] for from_loc in local_customers_var]) - lpSum(
            [assignment_var[customer, to_loc] for to_loc in local_customers_var]) == 0, "forTrip" + str(
            customer)

    # Each vehicle should enter a depot
    print('Each vehicle should enter a depot')
    sub_model += lpSum([assignment_var[customer, depot]
                            for customer in local_customers_var]) == 1, "exitDepotConnection"

    # vehicle Capacity
    print('vehicle Capacity')
    sub_model += lpSum(
        [float(local_demands[from_loc]) * assignment_var[from_loc, to_loc]
            for from_loc, to_loc in local_assignment_var_dict.keys()]) <= float(truck_capacity), "Capacity"

    # Time intervals
    print('time intervals')
    for from_loc, to_loc in local_assignment_var_dict.keys():
        if to_loc == depot: continue
        stop_time = local_service_time[from_loc]
        sub_model += time_var[to_loc] - time_var[from_loc] >= \
                        stop_time + bigm * assignment_var[
                            from_loc, to_loc] - bigm, "timewindow" + str(
            from_loc) + 'p' + str(to_loc)

    # Time Windows
    print('time windows')
    for vertex in local_customers_var:
        time_var[vertex].bounds(float(local_earliest_start[vertex]),
                                float(local_latest_end[vertex]))

    if lp_file_name is not None:
        sub_model.writeLP('{}.lp'.format(str(lp_file_name)))

    print("Using solver ", solver_type)
    if solver_type == 'PULP_CBC_CMD':
        sub_model.solve(PULP_CBC_CMD(
            msg=enable_solution_messaging,
            timeLimit=60 * solver_time_limit_minutes,
            fracGap=mip_gap)
        )
    elif solver_type == "GUROBI_CMD":
        solver = getSolver('GUROBI_CMD', msg=enable_solution_messaging,
            timeLimit=60 * solver_time_limit_minutes)
        sub_model.solve(solver)

    if LpStatus[sub_model.status] in ('Optimal', 'Undefined'):
        print('Sub Model Status = {}'.format(LpStatus[sub_model.status]))
        print("Sub model optimized objective function= ", value(sub_model.objective))
        solution_objective = value(sub_model.objective)
        # get assignment variable values
        #print('getting solution for assignment variables')
        route_dict = {}
        for from_loc, to_loc in local_assignment_var_dict.keys():
            if assignment_var[from_loc, to_loc].value() > 0:
                route_dict[from_loc] = to_loc
        route = []
        cur_node = depot
        route_data = []
        print(route_dict)
        i = 0
        while i < nb_customers:
            if cur_node != depot:
                node = selected_customers[int(cur_node.split('_')[1])-1]
                if node in route: break
                else: route.append(node)
            route_data.append([cur_node, route_dict[cur_node], local_demands[cur_node], time_var[cur_node].value(), local_earliest_start[cur_node], local_latest_end[cur_node]])
            cur_node = route_dict[cur_node]
            i += 1
            if cur_node == depot:
                break
        route_df = pd.DataFrame(data=route_data, columns=['previous_node', "next_node", "demand", "arrive_time", "ready_time", "due_time"])
        return solution_objective, route, route_df
    else:
        print('Model Status = {}'.format(LpStatus[sub_model.status]))
        raise Exception('No Solution Exists for the Sub problem')

def add_path(route, paths_dict, paths_cost_dict, paths_customers_dict, nb_customers, distance_warehouses, distance_matrix):
    total_num_path = len(paths_dict.keys())
    path_name = f"PATH_{total_num_path}"
    paths_dict[path_name] = [f"Customer_{c+1}" for c in route]
    paths_cost_dict[path_name] = distance_warehouses[route[0]] + distance_warehouses[route[-1]]
    for j in range(len(route)-1):
        paths_cost_dict[path_name] += distance_matrix[route[j]][route[j+1]]
    for j in range(nb_customers):
        paths_customers_dict[path_name, f"Customer_{j+1}"] = 0
    for j in route:
        paths_customers_dict[path_name, f"Customer_{j+1}"] = 1
    return paths_dict, paths_cost_dict, paths_customers_dict

import os
if __name__ == '__main__':
    problem_file = "/home/lesong/cvrptw/cvrp_benchmarks/homberger_100_customer_instances/c104.txt"
    dir_name = os.path.dirname(problem_file)
    file_name = os.path.splitext(os.path.basename(problem_file))[0]
    (nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_warehouses, demands, service_time,
                earliest_start, latest_end, max_horizon, warehouse_x, warehouse_y, customers_x, customers_y) = read_input_cvrptw(problem_file)
    num_episodes = 100
    min_path_frag_len, max_path_frag_len = 5, 15
    if True:
        print("solving a non-constraint tsp problem")
        tsp_solution = get_tsp_solution(nb_customers, distance_warehouses, distance_matrix)
        total_num_path = nb_customers
        paths_dict = {}
        paths_cost_dict = {}
        paths_customers_dict = {}
        for i in range(nb_customers):
            path_name = f"PATH_{i}"
            customer = f"Customer_{i+1}"
            paths_dict[path_name] = [customer]
            paths_cost_dict[path_name] = distance_warehouses[i]*2
            for j in range(nb_customers):
                paths_customers_dict[path_name, f"Customer_{j+1}"] = 0
            paths_customers_dict[path_name, customer] = 1
        ## initialize path from tsp
        num_selected_customers = 0
        for i in range(3):
            if i == 0: _tsp_solution = tsp_solution[:]
            else:
                idx = np.random.randint(1, nb_customers-1)
                _tsp_solution = tsp_solution[idx:] + tsp_solution[:idx]
            while len(_tsp_solution) > 0:
                print(f"selected customers {num_selected_customers}")
                path_frag_len = np.random.randint(min_path_frag_len, max_path_frag_len)
                selected_customers = _tsp_solution[:min(path_frag_len, len(_tsp_solution))]
                solution_objective, route, route_df = cvrptw_one_vehicle(selected_customers, truck_capacity, distance_matrix, distance_warehouses, demands, service_time,
                                        earliest_start, latest_end, max_horizon, solver_type="PULP_CBC_CMD")
                paths_dict, paths_cost_dict, paths_customers_dict = add_path(route, paths_dict, paths_cost_dict, paths_customers_dict, nb_customers, distance_warehouses, distance_matrix)
                num_selected_customers += len(route)
                for c in route:
                    _tsp_solution.remove(c)
        print(f"selected customers {num_selected_customers}")

        with open(f"{dir_name}/{file_name}_path.txt", "wb") as f:
            pickle.dump(paths_dict, f)
        with open(f"{dir_name}/{file_name}_path_cost.txt", "wb") as f:
            pickle.dump(paths_cost_dict, f)
        with open(f"{dir_name}/{file_name}_path_customer.txt", "wb") as f:
            pickle.dump(paths_customers_dict, f)
    else:
        with open(f"{dir_name}/{file_name}_path.txt", "rb") as f:
            paths_dict = pickle.load(f)
        with open(f"{dir_name}/{file_name}_path_cost.txt", "rb") as f:
            paths_cost_dict = pickle.load(f)
        with open(f"{dir_name}/{file_name}_path_customer.txt", "rb") as f:
            paths_customers_dict = pickle.load(f)
        

    # fine tuning using dual-variable
    for i in range(num_episodes):
        print(f"master model, episode: {i}")
        total_cost, prices_dict = path_selection(nb_customers, 
                                            paths_dict,
                                            paths_cost_dict,
                                            paths_customers_dict,
                                            solver_type='PULP_CBC_CMD')
        print("total cost: ", total_cost)
        print(f"sub model, episode: {i}")
        all_customers = list(range(nb_customers))
        prices = list(prices_dict.values())
        min_price, max_price = np.min(prices), np.max(prices)
        norm_prices = [(x-min_price)/max_price for x in prices]
        total_norm_price = np.sum(norm_prices)
        prob_prices = [x/total_norm_price for x in norm_prices]
        path_frag_len = np.random.randint(min_path_frag_len, max_path_frag_len)
        selected_customers = np.random.choice(all_customers, size=path_frag_len, replace=False, p=prob_prices)
        solution_objective, route, route_df = cvrptw_one_vehicle(selected_customers, truck_capacity, distance_matrix, distance_warehouses, demands, service_time,
                                earliest_start, latest_end, max_horizon, prices=prices_dict, solver_type="PULP_CBC_CMD")
        print("objective: ", solution_objective, 'route: ', route)
        print("capacity: ", truck_capacity, "total demand: ", route_df["demand"].sum())
        paths_dict, paths_cost_dict, paths_customers_dict = add_path(route, paths_dict, paths_cost_dict, paths_customers_dict, nb_customers, distance_warehouses, distance_matrix)