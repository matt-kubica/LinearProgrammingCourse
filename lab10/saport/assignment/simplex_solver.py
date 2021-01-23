import numpy as np
from .model import AssignmentProblem, Assignment, NormalizedAssignmentProblem
from ..simplex.model import Model
from ..simplex.expressions.expression import Expression
from dataclasses import dataclass
from typing import List 



class Solver:
    '''
    A simplex solver for the assignment problem.

    Methods:
    --------
    __init__(problem: AssignmentProblem):
        creates a solver instance for a specific problem
    solve() -> Assignment:
        solves the given assignment problem
    '''
    def __init__(self, problem: AssignmentProblem):
        self.problem = NormalizedAssignmentProblem.from_problem(problem)
        
    def solve(self) -> Assignment:
        # initialize model
        model = Model("assignment")

        # create variables, one for each cost in the cost matrix
        # add constraint, that every variable has to be <= 1
        # form objective expression - all variables with their cost factor
        objective = Expression()
        for row_index in range(self.problem.costs.shape[0]):
            for col_index in range(self.problem.costs.shape[1]):
                x = model.create_variable('x{0}{1}'.format(row_index, col_index))
                objective += (self.problem.costs[row_index, col_index] * x)
                model.add_constraint(x <= 1)

        # adding constraint, that sum of every row has to be equal 1
        for row_index in range(self.problem.costs.shape[0]):
            expression = Expression()
            for col_index in range(self.problem.costs.shape[1]):
                expression += [var for var in model.variables
                               if var.name == 'x{0}{1}'.format(row_index, col_index)][0]
            model.add_constraint(expression == 1)

        # adding constraint, that sum of every column has to be equal 1
        for col_index in range(self.problem.costs.shape[1]):
            expression = Expression()
            for row_index in range(self.problem.costs.shape[0]):
                expression += [var for var in model.variables
                               if var.name == 'x{0}{1}'.format(row_index, col_index)][0]
            model.add_constraint(expression == 1)

        # add objective to model, solve model
        model.minimize(objective)
        solution = model.solve()


        original = self.problem.original_problem.costs
        costs = self.problem.costs
        assigned_tasks = [-1] * original.shape[0]
        for worker in range(original.shape[0]):
            tasks = solution.assignment[costs.shape[1] * worker : (costs.shape[1] * worker) + costs.shape[1]]
            task = tasks.index(max(tasks))
            if task < original.shape[1]:
                assigned_tasks[worker] = task

        total_cost = 0
        for worker, task in enumerate(assigned_tasks):
            if task < original.shape[1] and task >= 0:
                c = original[worker, task]
                total_cost += c if c >= 0 else 0

        # print('simplex = {0}'.format(assigned_tasks))
        return Assignment(assigned_tasks, total_cost)



