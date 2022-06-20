import numpy as np
import elkai
from cvrptw import read_input_cvrptw

def get_tsp_solution(nb_customers, distance_warehouses, distance_matrix):
    M = np.zeros((nb_customers+1, nb_customers+1))
    for i in range(nb_customers+1):
        for j in range(nb_customers+1):
            if i == 0 and j == 0: continue
            elif i == 0: M[i, j] = distance_warehouses[j-1]
            elif j == 0: M[i, j] = distance_warehouses[i-1]
            else: M[i, j] = distance_matrix[i-1][j-1]
    route = elkai.solve_float_matrix(M)[1:]
    return [c-1 for c in route]


if __name__ == '__main__':
    problem_file = "/data/songlei/cvrptw-optimization/cvrp_benchmarks/homberger_400_customer_instances/C1_4_2.TXT"
    (nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_warehouses, demands, service_time,
                earliest_start, latest_end, max_horizon, warehouse_x, warehouse_y, customers_x, customers_y) = read_input_cvrptw(problem_file)

    print(get_tsp_solution(nb_customers, distance_warehouses, distance_matrix))
