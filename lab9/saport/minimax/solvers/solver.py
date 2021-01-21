from ..model import Game, Equilibrium
from typing import List 

class AbstractSolver:

    def __init__(self, game: Game):
        self.game = game

    def solve(self) -> List[Equilibrium]:
        raise Exception("This is an abstract solver, don't call it directly!")