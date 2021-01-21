from .greedy import AbstractGreedySolver
from ..model import Item

class GreedySolverDensity(AbstractGreedySolver):
    """
    A greedy solver for the knapsack problems. 
    Uses value/weight density as the greedy heuristic. 
    """
    def greedy_heuristic(self, item: Item) -> float:
        return item.value / item.weight