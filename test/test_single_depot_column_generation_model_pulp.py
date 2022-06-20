'''
Test class for testing column generation model
'''

import os
import sys
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'cvrptw_optimization')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'cvrptw_optimization/src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'cvrptw_optimization/data')))
from cvrptw_optimization.data import data as dat

# depots = dat.depots_unit_test
# customers = dat.customers_unit_test
# transportation_matrix = dat.transportation_matrix_unit_test
# vehicles = dat.vehicles_unit_test.head(2)


depots = dat.depots1
customers = dat.customers1
transportation_matrix = dat.transportation_matrix1
vehicles = dat.vehicles1.head(15)
capacity = vehicles.iloc[0, :]['CAPACITY']


print("depots:", depots)
print("customers: ", customers)
print("transportation_matrix: ", transportation_matrix)
print("vehicles: ", vehicles)


class SingleDepotTest(unittest.TestCase):

    def test_single_depot_column_generation_model_pulp(self):
        '''
        Test for the single depot problem
        :return:
        '''

        from cvrptw_optimization.src import single_depot_column_generation_pulp_inputs as inputs
        from cvrptw_optimization.src import single_depot_column_generation_pulp_problem_formulation as formulation
        from cvrptw_optimization.single_depot_column_generation_pulp import run_single_depot_column_generation

        solution, solution_statistics = run_single_depot_column_generation(
                                            depots,
                                            customers,
                                            transportation_matrix,
                                            vehicles,
                                            capacity,
                                            mip_gap=0.001,
                                            solver_time_limit_minutes=10,
                                            enable_solution_messaging=0,
                                            solver_type='PULP_CBC_CMD',
                                            max_iteration=50)
        print(solution)
        print(solution_statistics)
        for path_name, route in solution.groupby("PATH_NAME"):
            location_list = route.sort_values(by="STOP_NUMBER")["LOCATION_NAME"].tolist()
            print(path_name, location_list)

if __name__ == '__main__':
    unittest.main()
