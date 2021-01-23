from .bnb import AbstractBnbSolver
from ..model import Problem, Solution, Item
from typing import List 

class BnbDFSSolver(AbstractBnbSolver):
    """
    A branch-and-bound solver for the Knapsach Problem, 
    explores the search tree using a basic DFS strategy.
    """
    def dfs_bnb(self):
        self.best_solution = Solution.empty()
        return self._dfs_bnb(self.problem.items, self.best_solution)

    def _dfs_bnb(self, left: List[int], solution: Solution):
        if len(left) == 0:
            if solution.value > self.best_solution.value:
                self.best_solution = solution
            return

        if self.timeout():
            self.interrupted = True
            return 

        space_left = self.problem.capacity - solution.weight
        item = left[0]
        new_left = left[1:]
        if item.weight <= space_left:
            new_solution = solution.with_added_item(item)
            if self.upper_bound(new_left, new_solution) > self.best_solution.value:
                self._dfs_bnb(new_left, new_solution)

        if self.upper_bound(new_left, solution) > self.best_solution.value:
            self._dfs_bnb(new_left, solution)

    def solve(self) -> Solution:
        self.interrupted = False
        self.start_timer()
        self.dfs_bnb()
        self.best_solution.optimal = not self.interrupted
        self.stop_timer()
        return self.best_solution