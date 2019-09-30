"""
The Knapsack Problem solver from
https://developers.google.com/optimization/bin/knapsack
"""
import sys
import argparse
from ortools.algorithms import pywrapknapsack_solver as knapsack
from itertools import product
import  numpy as np


class Knapsack(object):
    """
    Class knapsack is a wrapper from Google's or-tools to solve the knapsack
    problem, to pack a set of items, with given sizes and values, into a
    container with a fixed capacity, so as to maximize the total value of the
    packed items
    """
    def __init__(self, items_names, values, weights, capacity, solve_type=5,
                 name='KnapsackExample'):
        """
        Constructor of class knapsack
        :param items_names: Vector with the items' names (match values)
        :param values: A vector containing the values of the items
        :param weights: A vector containing the weights of the items
        :param capacity: A numerical value of the capacity of the knapsack
        :param name: Name for the solver
        :param solve_type: Integer pointing to a type of optimization (below)

        solvertypes:
        0: KNAPSACK_BRUTE_FORCE_SOLVER
        1: KNAPSACK_64ITEMS_SOLVER
        2: KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER
        3: KNAPSACK_MULTIDIMENSION_CBC_MIP_SOLVER
        5: KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER
        """
        self.value_factor = 1
        self.weight_factor = 1
        self.items_names = items_names
        self.values = values
        self.weights = weights
        self.name = name
        self.solver_type = solve_type
        self.knapsack_solver = knapsack.KnapsackSolver
        self.solver = name
        self.capacity = capacity
        self.result = None
        self.packed_items = []
        self.packed_weights = []
        self.total_weight = 0

    @property
    def values(self):
        return self.__values

    @values.setter
    def values(self, values):
        for x in values:
            if isinstance(x, float):
                self.value_factor = 1000
                break
        self.__values = list((np.array(values) * self.value_factor).astype(int)
                             )

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, weights):
        for x in weights:
            if isinstance(x, float):
                self.weight_factor = 1000
                break
        self.__weights = [list((np.array(weights) * self.weight_factor).astype(
            int))]

    @property
    def solver(self):
        return self.__solver

    @solver.setter
    def solver(self, name):
        """
        Execute the solver
        """
        self.__solver = self.knapsack_solver(self.solver_type, name)

    @property
    def capacity(self):
        return self.__capacity

    @capacity.setter
    def capacity(self, capacity):
        self.__capacity = capacity * self.weight_factor
        self.result = None
        self.packed_items = []
        self.packed_weights = []
        self.total_weight = 0

    def solve(self):
        self.solver.Init(self.values, self.weights, [self.capacity])
        self.result = self.solver.Solve() / self.value_factor

    def get_results(self, print_it=False):
        """
        Populate human readable results
        :return:
        """
        self.solve()
        for i, name in enumerate(self.items_names):
            if self.solver.BestSolutionContains(i):
                weight = self.weights[0][i] / self.weight_factor
                self.packed_items.append(name)
                self.packed_weights.append(weight)
                self.total_weight += weight
        assert self.total_weight == sum(self.packed_weights)
        if print_it:
            print('Total value:', self.result)
            print('Total weight:', self.total_weight)
            print('Top 10 Packed items:', self.packed_items[:10])
            print('Top 10 Packed weights:', self.packed_weights[:10])


def test():
    """
    Dummy test, to include or-tools example. Eventually will be converted to
    unittest
    :return:
    """
    items_names = ['architecture training', 'enable', 'operations', 'define',
                   'cells', 'host', 'market', 'cases', 'custom', 'end',
                   'benchmark', 'action', 'update', 'format', 'platforms',
                   'basic', 'aims', 'transfer', 'international', 'project',
                   'square', 'latest', 'pytorch', 'multiplying', 'current',
                   'benchmarking', 'installation', 'environment',
                   'organization', 'parts', 'optimization', 'announcement',
                   'theory', 'accessing', 'short', 'debugging', 'bit',
                   'reached', 'benefits', 'stages', 'annealing',
                   'demonstration', 'downloaded', 'areas', 'reliable',
                   'worlds', 'rdp', 'setting', 'detector', 'cpus']

    values = [1.483, 6.133, 1.889, 3.285, 5.75, 1.544, 1.400, 1.676, 2.00,
              0.368, 1.678, 2.325, 2.530, 1.982, 1.870, 3.00, 3.833, 4.312,
              2.615, 1.434, 0.844, 9.842, 1.497, 1.611, 0.466, 1.182, 1.910,
              1.541, 2.844, 3.373, 1.358, 11.75, 0.306, 23.5, 1.235, 2.5,
              2.828, 3.357, 2.333, 1.172, 2.344, 1.371, 3.833, 4.0, 1.24,
              1.029, 1.835, 4.536, 6.275, 5.5]

    weights = [0.31, 0.15, 2.27, 0.14, 0.08, 25.53, 8.51, 4.2, 1.8, 5.86,
               34.93, 0.4, 1.64, 56.89, 1.24, 0.31, 0.12, 0.32, 2.0, 25.8,
               1195.44, 0.19, 19.48, 0.9, 12.23, 9.57, 4.9, 9.47, 5.22, 5.79,
               1.34, 0.04, 442.07, 0.02, 9.37, 0.18, 0.35, 0.14, 7.92, 28.94,
               3.89, 0.35, 0.12, 0.12, 1.5, 1.35, 46.9, 0.69, 0.29, 0.08]
    capacities = 2.0
    ks = Knapsack(items_names=items_names, values=values, weights=weights,
                  capacity=capacities)
    ks.get_results(print_it=True)
    print(sum(ks.packed_weights))


def main(items_names=None, values=None, weights=None, capacity=None,
         name='KnapsackExample', solver_type=5):
    """
    Execute the script
    :param values:
    :param weights:
    :param capacity:
    :param name:
    :param solver_type:
    """
    if values is None:
        # just test
        test()
        sys.exit()

    ks = Knapsack(items_names, values, weights, capacity, solver_type, name)
    ks.get_results()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PROG', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--items_names', default=None,
                        help='Names of items to optimize')
    parser.add_argument('-v', '--values', default=None,
                        help='Value/importance of the items')
    parser.add_argument('-w', '--weights', default=None,
                        help='Weight/cost of the items')
    parser.add_argument('-c', '--capacity', default=None,
                        help='Capacity of the knapsack')
    parser.add_argument('-n', '--name', default='KnapsackExample',
                        help='Solver name')
    parser.add_argument('-s', '--solver_type', default=5, type=int,
                        choices=[0, 1, 2, 3, 5],
                        help='Type of solver. One of '
                             '0: KNAPSACK_BRUTE_FORCE_SOLVER, '
                             '1: KNAPSACK_64ITEMS_SOLVER, '
                             '2: KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER, '
                             '3: KNAPSACK_MULTIDIMENSION_CBC_MIP_SOLVER, '
                             '5: KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLV'
                             'ER')

    args = parser.parse_args()
    main(items_names=args.items_names, values=args.values,
         weights=args.weights, capacity=args.capacity, name=args.name,
         solver_type=args.solver_type)