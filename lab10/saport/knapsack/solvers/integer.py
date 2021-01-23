from ..abstractsolver import AbstractSolver
from ..model import Problem, Solution, Item
from typing import List 
from ...integer.model import Model
from ...simplex.expressions.expression import Expression

class IntegerSolver(AbstractSolver):
    """
    An Integer Programming solver for the knapsack problems

    Methods:
    --------
    create_model() -> Models:
        creates and returns an integer programming model based on the self.problem
    """

    def create_model(self) -> Model:
        m = Model('knapsack')
        vars = [m.create_variable(f"x{i}") for i in self.problem.items]
        weights = [item.weight for item in self.problem.items]
        values = [item.value for item in self.problem.items]
        m.maximize(Expression.from_vectors(vars, values))
        m.add_constraint(Expression.from_vectors(vars, weights) <= self.problem.capacity)
        for v in vars:
            m.add_constraint(v <= 1)
        return m
    
    def solve(self) -> Solution:
        m = self.create_model()
        integer_solution = m.solve(self.timelimit)
        items = [item for (i,item) in enumerate(self.problem.items) if integer_solution.value(m.variables[i]) > 0]
        solution = Solution.from_items(items, not m.solver.interrupted)
        self.total_time = m.solver.total_time
        return solution