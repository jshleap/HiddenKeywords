"""
The Knapsack Problem solver from
https://developers.google.com/optimization/bin/knapsack
"""
from ortools.algorithms import pywrapknapsack_solver as knapsack



class knapsack(object):
    def __init__(self, values, weights, capacities, name='KnapsackExample',
                 solve_type=5):
        # solvertypes:
        # 0: KNAPSACK_BRUTE_FORCE_SOLVER
        # 1: KNAPSACK_64ITEMS_SOLVER
        # 2: KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER
        # 3: KNAPSACK_MULTIDIMENSION_CBC_MIP_SOLVER
        # 5: KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER

        self.values = values
        self.weights = weights
        self.capacities = capacities
        self.name = name
        self.solver_type = solve_type
        self.knapsack_solver = knapsack.KnapsackSolver
        self.solver = self.knapsack_solver(self.solver_type, name)
        self.result = None

    def solve(self):
        self.solver.Init(self.values, self.weights, self.capacities)
        self.result = self.solver.Solve()

    def test(self):
        values = [
            360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48,
            147, 78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
            514, 28, 87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10,
            19, 389, 276,312]
        weights = [[ 7, 0, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 0, 36, 3, 8,
                     15, 42, 9, 0, 42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4,
                     18, 56, 7, 29, 93, 44, 71, 3, 86, 66, 31, 65, 0, 79, 20,
                     65, 52, 13]]
        self.capacities = [850]
        self.solve()
        packed_items = []
        packed_weights = []
        total_weight = 0
        print('Total value =', self.result)
        for i in range(len(values)):
            if self.solver.BestSolutionContains(i):
                packed_items.append(i)
                packed_weights.append(weights[0][i])
                total_weight += weights[0][i]
        print('Total weight:', total_weight)
        print('Packed items:', packed_items)
        print('Packed_weights:', packed_weights)


if __name__ == '__main__':
    # TODO: add options and a main function
    pass