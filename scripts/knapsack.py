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
        self.capacity = capacity
        self.name = name
        self.solver_type = solve_type
        self.knapsack_solver = knapsack.KnapsackSolver
        self.solver = name
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
        self.__capacity = [capacity * self.weight_factor]
    def solve(self):
        self.solver.Init(self.values, self.weights, self.capacity)
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
        if print_it:
            print('Total value =', self.result)
            print('Total weight:', self.total_weight)
            print('Packed items:', self.packed_items)
            print('Packed weights:', self.packed_weights)


def test():
    """
    Dummy test, to include or-tools example. Eventually will be converted to
    unittest
    :return:
    """
    items_names = ['architecture training', 'enable', 'operations', 'define', 'cells',
     'host', 'market', 'cases', 'custom', 'end', 'benchmark', 'action',
     'update', 'format', 'platforms', 'basic', 'aims', 'transfer',
     'international', 'project', 'square', 'latest', 'pytorch', 'multiplying',
     'current', 'benchmarking', 'installation', 'environment', 'organization',
     'parts', 'optimization', 'announcement', 'theory', 'accessing', 'short',
     'debugging', 'bit', 'reached', 'benefits', 'stages', 'annealing',
     'demonstration', 'downloaded', 'areas', 'reliable', 'worlds', 'rdp',
     'setting', 'detector', 'cpus']
    values = [1.4838709677419357, 6.133333333333334, 1.8898678414096917,
     3.2857142857142856, 5.75, 1.54406580493537, 1.400705052878966,
     1.676190476190476, 2.0055555555555555, 0.36860068259385664,
     1.67849985685657, 2.325, 2.530487804878049, 1.9827737739497275,
     1.8709677419354842, 3.0000000000000004, 3.833333333333334, 4.3125, 2.615,
     1.4348837209302328, 0.8444171183831894, 9.842105263157896,
     1.4974332648870634, 1.6111111111111114, 0.4668847097301717,
     1.1828631138975967, 1.9102040816326527, 1.5417106652587116,
     2.844827586206897, 3.373056994818653, 1.3582089552238805, 11.75,
     0.30655778496618186, 23.5, 1.2358591248665955, 2.5, 2.8285714285714287,
     3.3571428571428568, 2.333333333333333, 1.1720801658604005,
     2.344473007712082, 1.3714285714285714, 3.833333333333334, 4.0, 1.24,
     1.0296296296296297, 1.8353944562899787, 4.536231884057972,
     6.275862068965518, 5.5]
    weights = [0.31, 0.15, 2.27, 0.14, 0.08, 25.53, 8.51, 4.2, 1.8, 5.86, 34.93, 0.4,
     1.64, 56.89, 1.24, 0.31, 0.12, 0.32, 2.0, 25.8, 1195.44, 0.19, 19.48, 0.9,
     12.23, 9.57, 4.9, 9.47, 5.22, 5.79, 1.34, 0.04, 442.07, 0.02, 9.37, 0.18,
     0.35, 0.14, 7.92, 28.94, 3.89, 0.35, 0.12, 0.12, 1.5, 1.35, 46.9, 0.69,
     0.29, 0.08]
    # values = [
    #     360.5, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48,
    #     147, 78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
    #     514, 28, 87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10,
    #     19, 389, 276, 312]
    # items_names = [''.join([x[0], str(x[1])]) for x in product(['A'], values)]
    # weights = [7, 0, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 0, 36, 3, 8,
    #              15, 42, 9, 0, 42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4,
    #              18, 56, 7, 29, 93, 44, 71, 3, 86, 66, 31.5, 65, 0, 79, 20,
    #              65, 52, 13]
    capacities = 20#850
    ks = Knapsack(items_names=items_names, values=values, weights=weights,
                  capacity=capacities)
    ks.solve()
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