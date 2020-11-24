from ..abstractsolver import AbstractSolver
from ..model import Problem, Solution, Item
from typing import List


class AbstractBnbSolver(AbstractSolver):
    """
    An abstract branch-and-bound solver for the knapsack problems.

    Methods:
    --------
    upper_bound(left : List[Item], solution: Solution) -> float:
        given the list of still available items and the current solution,
        calculates the linear relaxation of the problem
    """

    def upper_bound(self, left: List[Item], solution: Solution) -> float:
        actual_weight = solution.weight
        capacity = self.problem.capacity
        upper_bound = solution.value
        for item in left:
            if actual_weight >= capacity:
                break
            free_weight = capacity - actual_weight
            item_fraction = min(1, free_weight / item.weight)
            actual_weight += item_fraction * item.weight
            upper_bound += item_fraction * item.value
        return upper_bound

    def solve(self) -> Solution:
        raise Exception("this is an abstract solver, don't try to run it!")
