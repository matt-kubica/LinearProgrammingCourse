from dataclasses import dataclass
from .solver import AbstractSolver
from ..model import Game, Equilibrium, Strategy
from typing import List

class PureSolver(AbstractSolver):

    def solve(self) -> List[Equilibrium]:
        table = self.game.reward_matrix
        a_num_actions, _ = table.shape
        a_actions = range(a_num_actions)

        maximin_actions = [row.argmin() for row in table]
        minimax_actions = [col.argmax() for col in table.T]

        return [self.create_equilibrium(a, maximin_actions[a]) for a in a_actions if a == minimax_actions[maximin_actions[a]]]

    def create_equilibrium(self, a, b):
        a_num_actions, b_num_actions = self.game.reward_matrix.shape
        return Equilibrium(self.game.reward_matrix[a,b],
            Strategy.with_action(a, a_num_actions),
            Strategy.with_action(b, b_num_actions))








