from numpy.core.arrayprint import format_float_positional
from ..abstractsolver import AbstractSolver
from ..model import Problem, Solution, Item
import time


class AbstractGreedySolver(AbstractSolver):
    """
    An abstract greedy solver for the knapsack problems.

    Methods:
    --------
    greedy_heuristic(item : Item) -> float:
        return a value representing how much the given items is valuable to the greedy algorithm
        bigger value > earlier to take in the backpack
    """

    def greedy_heuristic(self, item: Item) -> float:
        raise Exception("Greedy solver requires a heuristic!")

    def solve(self) -> Solution:
        self.start_timer()
        sorted_items = sorted(self.problem.items, key=self.greedy_heuristic)
        choosen_items = []
        now_weight = 0
        now_value  = 0
        while len(sorted_items) > 0 and now_weight <= self.problem.capacity:
            item = sorted_items.pop()
            if item.weight + now_weight <= self.problem.capacity:
                choosen_items.append(item)
                now_weight += item.weight
                now_value += item.value

        self.stop_timer()
        return Solution(items=choosen_items, value=now_value, weight=now_weight, optimal=False)