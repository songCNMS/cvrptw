from platform import node
from select import select
import numpy as np
import pandas as pd
from pulp import *
import sys
import pickle
from cvrptw import read_input_cvrptw
from tsp import get_tsp_solution
from cvrptw_single_route import path_selection, cvrptw_one_vehicle, add_path


depot = "Customer_0"

def construct_init_solution(all_customers, truck_capacity,
                            demands, service_time, 
                            earliest_start, latest_end,
                            distance_matrix):
    num_routes = np.sum(list(demands.values())) // truck_capacity
    cur_routes = {}
    selected_customers = []
    sorted_customers = sorted([(c, distance_matrix[depot][c]) for c in all_customers], reverse=True, key=lambda x: x[1])
    for i in range(1, 1+num_routes):
        cur_routes[f"PATH_NAME_{i}"] = [sorted_customers[i-1][0]]
        selected_customers.append(sorted_customers[i-1][0])
    for c in all_customers:
        if c in selected_customers: continue
        selected_customers.append(c)
        route_cost_list = []
        for route_name, route in cur_routes.items():
            if demands[c] + np.sum([demands[_c] for _c in route]) < truck_capacity:
                min_cost, min_pos = route_insertion_cost(route, c, service_time, 
                                                         earliest_start, latest_end,
                                                         distance_matrix)
                if min_pos is not None: route_cost_list.append((route_name, (min_cost, min_pos)))
        if len(route_cost_list) <= 0:
            num_routes += 1
            cur_routes[f"PATH_NAME_{num_routes}"] = [c]
        else:
            route_cost_list = sorted(route_cost_list, key=lambda x: x[1][0])
            route_name, insert_pos = route_cost_list[0][0], route_cost_list[0][1][1]
            cur_routes[route_name] = cur_routes[route_name][:insert_pos] + [c] + cur_routes[route_name][insert_pos:]
    return cur_routes


def select_candidate_points(routes, distance_matrix, all_customers):
    route_key = np.random.choice(list(routes.keys()))
    route = routes[route_key]
    # if len(route) < 3: return []
    if len(route) >= 3:
        node_idx = np.random.randint(0, len(route)-1)
        M = [route[node_idx], route[node_idx+1]]
        next_node = route[node_idx+1]
    else: 
        node_idx = 0
        M = route[:]
        next_node = depot
    dist = [(c, distance_matrix[route[node_idx]][c]+distance_matrix[c][next_node]) for c in all_customers if c not in M]
    min_c1, min_c2 = None, None
    min_dist1, min_dist2 = np.float("inf"), np.float("inf")
    for c, d in dist:
        if c in route: continue
        if d < min_dist1:
            min_dist2 = min_dist1
            min_c2 = min_c1
            min_dist1 = d
            min_c1 = c
        elif d < min_dist2:
            min_dist2 = d
            min_c2 = c
    M.extend([min_c1, min_c2])
    return M

def is_valid_pos(route, pos, customer, service_time, earliest_start, latest_end):
    new_route = route[:pos] + [customer] + route[pos:]
    cur_time = 0.0
    for r in [depot] + new_route + [depot]:
        if cur_time > latest_end[r]: return False
        cur_time = max(cur_time, earliest_start[r]) + service_time[r]
    return True

def route_insertion_cost(route, customer, service_time, 
                         earliest_start, latest_end,
                         distance_matrix):
    route_len = len(route)
    min_cost = np.float("inf")
    min_pos = None
    for i in range(route_len+1):
        if is_valid_pos(route, i, customer, service_time, earliest_start, latest_end):
            if i == 0:
                old_cost = distance_matrix[depot][route[0]]
                new_cost = distance_matrix[depot][customer] + distance_matrix[customer][route[0]]
            elif i == route_len:
                old_cost = distance_matrix[route[-1]][depot]
                new_cost = distance_matrix[customer][depot] + distance_matrix[route[-1]][customer]
            else:
                old_cost = distance_matrix[route[i-1]][route[i]]
                new_cost = distance_matrix[route[i-1]][customer] + distance_matrix[customer][route[i]]
            if new_cost - old_cost < min_cost: 
                min_cost = new_cost - old_cost
                min_pos = i 
    return min_cost, min_pos


def compute_route_cost(routes, distance_matrix):
    total_cost = 0.0
    for route in routes.values():
        total_cost += distance_matrix[depot][route[0]]
        for i in range(len(route)-1):
            total_cost += distance_matrix[route[i]][route[i+1]]
        total_cost += distance_matrix[route[-1]][depot]
    return total_cost


def heuristic_improvement(cur_routes, all_customers, truck_capacity, demands, service_time, 
                          earliest_start, latest_end,
                          distance_matrix):
    ori_total_cost = compute_route_cost(cur_routes, distance_matrix)
    customers = select_candidate_points(cur_routes, distance_matrix, all_customers)
    routes_before_insert = {}
    # print("ori routes: ", cur_routes)
    for route_name, route in cur_routes.items():
        new_route = [c for c in route if c not in customers]
        if len(new_route) > 0: routes_before_insert[route_name] = new_route
    # print("selected customers: ", customers)
    # print("after routes: ", routes_before_insert)
    total_cost_before_insert = compute_route_cost(routes_before_insert, distance_matrix)
    customer_to_route_dict = {}
    for c in customers:
        route_cost_list = sorted([(route_name, route_insertion_cost(route, c, service_time, 
                                  earliest_start, latest_end,
                                  distance_matrix))
                                  for route_name, route in routes_before_insert.items()], key=lambda x: x[1][0])
        customer_to_route_dict[c] = [x for x in route_cost_list[:min(2, len(route_cost_list))] if (x[1][1] is not None)]
    
    min_total_cost_increase = np.float("inf")
    new_routes_after_insertion = None
    for i in range(2**(len(customers))):
        idx_list = [(i//(2**j))%2 for j in range(len(customers))]
        if np.any([idx+1>len(customer_to_route_dict[c]) for idx, c in zip(idx_list, customers)]): continue
        customer_on_route = {}
        for idx, c in zip(idx_list, customers):
            route_name = customer_to_route_dict[c][idx][0]
            if route_name not in customer_on_route:
                customer_on_route[route_name] = [c]
            else: customer_on_route[route_name].append(c)
        valid_insertion = True
        total_cost_increase = 0.0
        routes_after_insertion = {}
        for route_name, customers_to_insert in customer_on_route.items():
            if not valid_insertion: break
            new_route = routes_before_insert[route_name][:]
            for c in customers_to_insert:
                if np.sum([demands[_c] for _c in new_route]) + demands[c] > truck_capacity: 
                    valid_insertion = False
                    break
                min_cost, min_pos = route_insertion_cost(new_route, c, service_time, 
                                                         earliest_start, latest_end,
                                                         distance_matrix)
                if min_pos is None:
                    valid_insertion = False
                    break
                else: 
                    new_route = new_route[:min_pos] + [c] + new_route[min_pos:]
                    total_cost_increase += min_cost
            routes_after_insertion[route_name] = new_route
        
        if valid_insertion and total_cost_increase < min_total_cost_increase:
            min_total_cost_increase = total_cost_increase
            new_routes_after_insertion = {route_name: route[:] for route_name, route in routes_after_insertion.items()}
    if min_total_cost_increase + total_cost_before_insert < ori_total_cost:
        new_routes = {}
        for route_name, route in routes_before_insert.items():
            if route_name in new_routes_after_insertion: new_routes[route_name] = new_routes_after_insertion[route_name]
            else: new_routes[route_name] = route
        # routes_before_insert.update(new_routes_after_insertion)
        # ori_total_cost = compute_route_cost(cur_routes, distance_matrix)
        ori_total_cost = min_total_cost_increase + total_cost_before_insert
    else: new_routes = cur_routes
    return new_routes, ori_total_cost


import matplotlib.pyplot as plt
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--problem", type=str)
parser.add_argument("--num_nodes", type=int)
parser.add_argument("--retrain", action="store_true")
parser.add_argument("--opt", action="store_true")
args = parser.parse_args()

if __name__ == '__main__':
    problem_file = f"/home/lesong/cvrptw/cvrp_benchmarks/homberger_{args.num_nodes}_customer_instances/{args.problem}.TXT"
    dir_name = os.path.dirname(problem_file)
    file_name = os.path.splitext(os.path.basename(problem_file))[0]
    (nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_warehouses, demands, service_time,
                earliest_start, latest_end, max_horizon, warehouse_x, warehouse_y, customers_x, customers_y) = read_input_cvrptw(problem_file)
    
    distance_matrix_dict = {}
    demands_dict = {}
    service_time_dict = {}
    earliest_start_dict = {}
    latest_end_dict = {}
    # tsp_solution = get_tsp_solution(nb_customers, distance_warehouses, distance_matrix)
    # all_customers = [f"Customer_{c+1}" for c in tsp_solution]
    all_customers = [f"Customer_{i}" for i in range(1, 1+nb_customers)]
    for i, customer1 in enumerate([depot] + all_customers):
        if i == 0:
            demands_dict[customer1] = 0
            service_time_dict[customer1] = 0
            earliest_start_dict[customer1] = 0
            latest_end_dict[customer1] = max_horizon
        else:
            demands_dict[customer1] = demands[i-1]
            service_time_dict[customer1] = service_time[i-1]
            earliest_start_dict[customer1] = earliest_start[i-1]
            latest_end_dict[customer1] = latest_end[i-1]
        distance_matrix_dict[customer1] = {}
        for j, customer2 in enumerate([depot] + all_customers):
            if i == 0 and j == 0: distance_matrix_dict[customer1][customer2] = 0.0
            elif i == 0 and j > 0: distance_matrix_dict[customer1][customer2] = distance_warehouses[j-1]
            elif i > 0 and j == 0: distance_matrix_dict[customer1][customer2] = distance_warehouses[i-1]
            else: distance_matrix_dict[customer1][customer2] = distance_matrix[i-1][j-1]
    
    num_episodes = 10000
    early_stop_rounds = 1000
    min_path_frag_len, max_path_frag_len = 8, 15
    if args.retrain:
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
        for i in range(1):
            if i == 0: _tsp_solution = tsp_solution[:]
            else:
                idx = np.random.randint(1, nb_customers-1)
                _tsp_solution = tsp_solution[idx:] + tsp_solution[:idx]
            while len(_tsp_solution) > 0:
                print(f"selected customers {num_selected_customers}")
                path_frag_len = 15 # np.random.randint(min_path_frag_len, max_path_frag_len)
                selected_customers = _tsp_solution[:min(path_frag_len, len(_tsp_solution))]
                solution_objective, route, route_df = cvrptw_one_vehicle(selected_customers, truck_capacity, distance_matrix, distance_warehouses, demands, service_time,
                                        earliest_start, latest_end, max_horizon, solver_type="PULP_CBC_CMD")
                paths_dict, paths_cost_dict, paths_customers_dict = add_path(route, paths_dict, paths_cost_dict, paths_customers_dict, nb_customers, distance_warehouses, distance_matrix)
                num_selected_customers += len(route)
                for c in route: _tsp_solution.remove(c)
        print(f"selected customers {num_selected_customers}")
        with open(f"{dir_name}/{file_name}_path.txt", "wb") as f:
            pickle.dump(paths_dict, f)
        with open(f"{dir_name}/{file_name}_path_cost.txt", "wb") as f:
            pickle.dump(paths_cost_dict, f)
        with open(f"{dir_name}/{file_name}_path_customer.txt", "wb") as f:
            pickle.dump(paths_customers_dict, f)
    if args.opt:
        with open(f"{dir_name}/{file_name}_path.txt", "rb") as f:
            paths_dict = pickle.load(f)
        with open(f"{dir_name}/{file_name}_path_cost.txt", "rb") as f:
            paths_cost_dict = pickle.load(f)
        with open(f"{dir_name}/{file_name}_path_customer.txt", "rb") as f:
            paths_customers_dict = pickle.load(f)
    
    cost_list = []
    if args.opt:
        total_cost, prices_dict, cur_routes = path_selection(nb_customers, 
                                                             paths_dict,
                                                             paths_cost_dict,
                                                             paths_customers_dict,
                                                             solver_type='PULP_CBC_CMD')
    else:
        cur_routes = construct_init_solution(all_customers, truck_capacity,
                                            demands_dict, service_time_dict, 
                                            earliest_start_dict, latest_end_dict,
                                            distance_matrix_dict)
        total_cost = compute_route_cost(cur_routes, distance_matrix_dict)
    print("Master model total cost: ", total_cost)
    cost_list.append(total_cost)
    # fine tuning using dual-variable
    for i in range(num_episodes):
        cur_routes, total_cost= heuristic_improvement(cur_routes, all_customers, truck_capacity, 
                                                      demands_dict, service_time_dict, 
                                                      earliest_start_dict, latest_end_dict,
                                                      distance_matrix_dict)
        print(f"Fine tune round {i}, total cost: {total_cost}")
        cost_list.append(total_cost)
        if len(cost_list) > early_stop_rounds and np.min(cost_list[-early_stop_rounds:]) >= np.min(cost_list[:-early_stop_rounds]):
            break
    plt.plot(cost_list)
    plt.ylabel('Total Distance')
    plt.savefig("total_cost.png")
    total_path_num = len(cur_routes.keys())
    total_nodes = set()
    for _, route in cur_routes.items():
        for c in route: total_nodes.add(c)
    print(f"total vehicles: {total_path_num}, total nodes: {len(total_nodes)}")
    print(cur_routes)