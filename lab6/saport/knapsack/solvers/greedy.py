from numpy.core.arrayprint import format_float_positional
from numpy.lib.function_base import append
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
        choosen_items = []
        now_weight = now_value = 0
        self.start_timer()
        items = sorted(self.problem.items, key=self.greedy_heuristic)
        for item in items:
            new_weight = now_weight + item.weight
            if new_weight <= self.problem.capacity:
                now_weight += item.weight
                now_value += item.value
                choosen_items.append(item)
        self.stop_timer()
        return Solution(items=choosen_items, value=now_value, weight=now_weight, optimal=False)
