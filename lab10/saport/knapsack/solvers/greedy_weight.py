from .greedy import AbstractGreedySolver
from ..model import Item

class GreedySolverWeight(AbstractGreedySolver):
    """
    A greedy solver for the knapsack problems. 
    Uses weight as the greedy heuristic. 
    """

    def greedy_heuristic(self, item: Item) -> float:
        return -item.weight