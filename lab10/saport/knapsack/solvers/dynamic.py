from numpy.lib.function_base import extract
from ..abstractsolver import AbstractSolver
from ..model import Problem, Solution, Item
from cachetools import cached, LRUCache
from cachetools.keys import hashkey
import time
import numpy as np
from typing import Tuple

class DynamicSolver(AbstractSolver):
    """
    A naive dynamic programming solver for the knapsack problem. 
    """
    def create_table(self) -> np.array:
        table = np.zeros((self.problem.capacity + 1, len(self.problem.items) + 1), int)

        for (i, item) in enumerate(self.problem.items):
            for c in range(self.problem.capacity + 1):
                if self.timeout():
                    return table
                if item.weight <= c:
                    table[c, i + 1] = max(table[c - item.weight, i] + item.value, table[c, i])
                else:
                    table[c, i + 1] = table[c, i]
        
        return table

    def extract_solution(self, table : np.array) -> Solution:
        used_items = []
        optimal = table[-1,-1] > 0
        current_weight = self.problem.capacity
        for (i, item) in reversed(list(enumerate(self.problem.items))):
            if table[current_weight, i+1] > table[current_weight, i]:
                used_items.append(item)
                current_weight -= item.weight

        return Solution.from_items(used_items, optimal)

    def solve(self) -> Tuple[Solution, float]:
        self.interrupted = False
        self.start_timer()
        
        table = self.create_table()
        solution = self.extract_solution(table)

        self.stop_timer()
        return solution