from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List

@dataclass
class Strategy:
    probabilities: List[float]

    @staticmethod
    def with_action(a: int, num_actions: int) -> Strategy:
        probabilities = [0] * num_actions
        probabilities[a] = 1.0
        return Strategy(probabilities)

    def __str__(self) -> str:
        return ','.join([f'{x:.2f}' for x in self.probabilities])

    def equals_enough(self, other: Strategy, eps = 0.001):
        if len(self.probabilities) != len(other.probabilities):
            return False
        
        for (i,p) in enumerate(self.probabilities):
            if abs(p - other.probabilities[i]) > eps:
                return False

        return True

@dataclass
class Equilibrium:
    value: float 
    strategy_a: Strategy
    strategy_b: Strategy

    def __str__(self) -> str:
        text = f"* game value: {self.value:.2f}\n"
        text += f"* Alice's strategy: {self.strategy_a}\n"
        text += f"* Bob's strategy: {self.strategy_b}"
        return text

    def equals_enough(self, other: Equilibrium, eps = 0.009) -> bool:
        return abs(self.value - other.value) <= eps and \
            self.strategy_a.equals_enough(other.strategy_a, eps) and \
            self.strategy_b.equals_enough(other.strategy_b, eps)


@dataclass
class Game:
    reward_matrix: np.array

    def __init__(self, reward_matrix: np.array):
        self.reward_matrix = reward_matrix

    @staticmethod
    def from_file(path: str) -> Game:
        with open(path) as f:
            header = list(map(int, f.readline().split()))
            n_actions_a = header[0]
            n_actions_b = header[1]
            reward_matrix = np.zeros((n_actions_a, n_actions_b), float)
            for i in range(n_actions_a):
                reward_row = list(map(float, f.readline().split()))
                for j in range(n_actions_b):
                    reward_matrix[i,j] = reward_row[j]
            
            return Game(reward_matrix)  

    def __str__(self) -> str:
        a_actions, b_actions = self.reward_matrix.shape        
        longest = max([max([len(f"{val:.2f}") for val in row]) for row in self.reward_matrix])
        text = "Alice vs Bob\n"
        lines = [" | ".join([f"{f'{x:.2f}': >{longest}}" for x in self.reward_matrix[a,:]]) for a in range(a_actions)]
        text += '\n'.join(lines)
        return text