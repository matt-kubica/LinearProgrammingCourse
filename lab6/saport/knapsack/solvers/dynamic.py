from ..abstractsolver import AbstractSolver
from ..model import Item, Solution
import numpy as np
from typing import Tuple


class DynamicSolver(AbstractSolver):
    """
    A naive dynamic programming solver for the knapsack problem. 
    """

    def create_table(self) -> np.array:
        if self.timeout():
            self.interrupted = True
            return
        table = np.zeros((self.problem.capacity + 1,
                          len(self.problem.items) + 1), int)
        for i, item in enumerate(self.problem.items, start=1):
            self._fill_column(table, i, item)
        return table

    def _fill_column(self, table, elem_ind: int, item: Item) -> None:
        for capacity in range(1, table.shape[0]):
            if item.weight > capacity:
                table[capacity, elem_ind] = table[capacity, elem_ind-1]
            else:
                with_new_item = table[capacity -
                                      item.weight, elem_ind - 1] + item.value
                without_new_item = table[capacity, elem_ind - 1]
                new_score = max(with_new_item, without_new_item)
                table[capacity, elem_ind] = new_score

    def extract_solution(self, table: np.array) -> Solution:
        used_items = []
        all_items = self.problem.items
        optimal = table[-1, -1] > 0
        row_ind = table.shape[0] - 1
        for col_ind in range(len(all_items)-1, 1, -1):
            if table[row_ind, col_ind] != table[row_ind, col_ind-1]:
                item = all_items[col_ind-1]
                used_items.append(item)
                row_ind -= item.weight
        return Solution.from_items(used_items, optimal)

    def solve(self) -> Tuple[Solution, float]:
        self.interrupted = False
        self.start_timer()
        table = self.create_table()
        solution = self.extract_solution(table)

        self.stop_timer()
        return solution
