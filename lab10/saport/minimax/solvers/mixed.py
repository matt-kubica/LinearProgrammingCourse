from dataclasses import dataclass
from .solver import AbstractSolver
from ..model import Game, Equilibrium, Strategy
from ...simplex import model as lpmodel
from ...simplex import solution as lpsolution
from ...simplex.expressions import expression as expr
import numpy as np
from typing import Tuple, List

class MixedSolver(AbstractSolver):

    def solve(self) -> Equilibrium:
        shifted_game, shift = self.shift_game_rewards()

        # don't remove this print, it will be graded :)
        print(f"- shifted game: \n{shifted_game}")

        a_model = self.create_max_model(shifted_game)
        b_model = self.create_min_model(shifted_game)       
        a_solution = a_model.solve()
        b_solution = b_model.solve()

        a_probabilities = self.extract_probabilities(a_solution)
        b_probabilities = self.extract_probabilities(b_solution)
        
        return Equilibrium(a_solution.objective_value() - shift, Strategy(a_probabilities), Strategy(b_probabilities))


    def shift_game_rewards(self) -> Tuple[Game, float]:
        maximin = max([min(row) for row in self.game.reward_matrix])
        shift = 0 if maximin >= 0 else -maximin
        return Game(self.game.reward_matrix + shift), shift

    def create_max_model(self, game: Game) -> lpmodel.Model:
        a_num_actions, _ = game.reward_matrix.shape
        a_actions = range(a_num_actions)
        a_model = lpmodel.Model("A")

        v = a_model.create_variable("v")
        xs = [a_model.create_variable(f"x{i}") for i in a_actions]
        
        a_model.add_constraint(expr.Expression.from_vectors(xs, [1 for _ in xs]) == 1)
        for col in game.reward_matrix.T:
            a_model.add_constraint(expr.Expression.from_vectors([v] + xs, [1] + list(-1 * col)) <= 0)
        a_model.maximize(v)

        return a_model

    def create_min_model(self, game: Game) -> lpmodel.Model:
        _, b_num_actions = game.reward_matrix.shape
        b_actions = range(b_num_actions)

        b_model = lpmodel.Model("B")
        v = b_model.create_variable("v")
        ys = [b_model.create_variable(f"y{i}") for i in b_actions]
        
        b_model.add_constraint(expr.Expression.from_vectors(ys, [1 for _ in ys]) == 1)
        for row in game.reward_matrix:
            b_model.add_constraint(expr.Expression.from_vectors([v] + ys, [1] + list(-1 * row)) >= 0)
        b_model.minimize(v)

        return b_model

    def extract_probabilities(self, solution: lpsolution.Solution) -> List[float]:
        return [solution.value(x) for x in solution.model.variables if not solution.model.objective.depends_on_variable(solution.model, x)]

