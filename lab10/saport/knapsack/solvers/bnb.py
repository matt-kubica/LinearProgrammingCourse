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
    
    def upper_bound(self, left : List[Item], solution: Solution) -> float:
        sorted_items = sorted(left, reverse=True, key=lambda i: i.value / i.weight)
        capacity_left = self.problem.capacity - solution.weight
        current_value = solution.value
        for item in sorted_items:
            how_much_to_take = min(capacity_left, item.weight)
            capacity_left -= how_much_to_take
            current_value += (how_much_to_take / item.weight) * item.value

            if capacity_left <= 0:
                break
        return current_value

        
    def solve(self) -> Solution:
        raise Exception("this is an abstract solver, don't try to run it!")