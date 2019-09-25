"""
The Knapsack Problem solver from
https://developers.google.com/optimization/bin/knapsack
"""
import sys
import argparse
from ortools.algorithms import pywrapknapsack_solver as knapsack
from itertools import product


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
        self.items_names = items_names
        self.values = values
        self.weights = [weights]
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
    def solver(self):
        return self.__solver

    @solver.setter
    def solver(self, name):
        """
        Execute the solver
        """
        self.__solver = self.knapsack_solver(self.solver_type, name)

    def solve(self):
        self.solver.Init(self.values, self.weights, self.capacity)
        self.result = self.solver.Solve()

    def get_results(self, print_it=False):
        """
        Populate human readable results
        :return:
        """
        self.solve()
        for i, name in enumerate(self.items_names):
            if self.solver.BestSolutionContains(i):
                self.packed_items.append(name)
                self.packed_weights.append(self.weights[0][i])
                self.total_weight += self.weights[0][i]
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
    values = [
        360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48,
        147, 78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323,
        514, 28, 87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10,
        19, 389, 276, 312]
    items_names = [''.join([x[0], str(x[1])]) for x in product(['A'], values)]
    weights = [7, 0, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 0, 36, 3, 8,
                 15, 42, 9, 0, 42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4,
                 18, 56, 7, 29, 93, 44, 71, 3, 86, 66, 31, 65, 0, 79, 20,
                 65, 52, 13]
    capacities = [850]
    ks = Knapsack(items_names=items_names, values=values, weights=weights,
                  capacity=capacities)
    ks.solve()
    ks.get_results(print_it=True)


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