from copy import deepcopy
import enum

from . import model as m 
from .expressions import objective as o 
from .expressions import constraint as c
from .expressions import variable as v
from . import solution as s 
from . import tableaux as t
import numpy as np 

class Solver:
    """
        A class to represent a simplex solver.

        Methods
        -------
        solve(model: Model) -> Solution:
            solves the given model and return the first solution
    """

    def solve(self, model):
        normal_model = self._normalize_model(model)
        if len(self.slack_variables) < len(normal_model.constraints):
            tableaux = self._presolve(normal_model)
        else:
            tableaux = self._basic_initial_tableaux(normal_model)

        initial_tableaux = deepcopy(tableaux)
        self._optimize(tableaux)
        assignment = tableaux.extract_assignment()
        return self._create_solution(assignment, model, initial_tableaux, tableaux, normal_model)

    def _optimize(self, tableaux):

        while not tableaux.is_optimal():
            pivot_col = tableaux.choose_entering_variable()
            if tableaux.is_unbounded(pivot_col):
                raise Exception("Linear Programming model is unbounded")
            pivot_row = tableaux.choose_leaving_variable(pivot_col)

            tableaux.pivot(pivot_row, pivot_col)

    def _presolve(self, model):
        """
            _presolve(model: Model) -> Tableaux:
                returns a initial tableaux for the second phase of simplex
        """
        presolve_model = self._create_presolve_model(model)
        tableaux = self._presolve_initial_tableaux(presolve_model)

        self._optimize(tableaux)

        
        if self._artifical_variables_are_positive(tableaux):
            raise Exception("Linear Programming model is unsolvable")

        basis = tableaux.extract_basis()
        tableaux = self._remove_artificial_variables(tableaux)
        tableaux = self._restore_original_objective_row(tableaux, model)
        tableaux = self._fix_objective_row_to_the_basis(tableaux, basis)
        return tableaux

    def _normalize_model(self, original_model):
        """
            _normalize_model(model: Model) -> Model:
                returns a normalized version of the given model 
        """

        model = original_model.translate_to_standard_form()
        self._change_constraints_bounds_to_nonnegative(model)
        self.slack_variables = self._add_slack_variables(model)
        self.surplus_variables = self._add_surplus_variables(model)   
        return model

    def _create_presolve_model(self, normalized_model):
        presolve_model = deepcopy(normalized_model)
        self.artificial_variables = self._add_artificial_variables(presolve_model)
        return presolve_model    

    def _change_constraints_bounds_to_nonnegative(self, model):
        for constraint in model.constraints:
            if constraint.bound < 0:
                constraint.invert()
    
    def _add_slack_variables(self, model):
        slack_variables = dict()
        for (i,constraint) in enumerate(model.constraints.copy()):
            if constraint.type == c.ConstraintType.LE:
                slack_var = model.create_variable(f"s{i}")
                slack_variables[slack_var] = i
                constraint.expression = constraint.expression + slack_var
                constraint.type = c.ConstraintType.EQ
        return slack_variables

    def _add_surplus_variables(self, model):
        surplus_variables = dict()
        for (i,constraint) in enumerate(model.constraints.copy()):
            if constraint.type == c.ConstraintType.GE:
                surplus_var = model.create_variable(f"s{i}")
                surplus_variables[surplus_var] = i
                constraint.expression = constraint.expression - surplus_var
                constraint.type = c.ConstraintType.EQ
        return surplus_variables 

    def _add_artificial_variables(self, model):
        artificial_variables = dict()
        for (i,constraint) in enumerate(model.constraints.copy()):
            if i in self.slack_variables.values():
                continue
            artificial_var = model.create_variable(f"R{i}")
            artificial_variables[artificial_var] = i
            constraint.expression = constraint.expression + artificial_var
        return artificial_variables

    def _presolve_initial_tableaux(self, model):
        objective_row = np.array([0.0 for _ in model.variables] + [0.0])

        for var in self.artificial_variables.keys():
            objective_row[var.index] = 1.0

        table = np.array([objective_row] + [c.expression.factors(model) + [c.bound] for c in model.constraints])

        for constraint_index in self.artificial_variables.values():
            constraint = model.constraints[constraint_index]
            factors_row = np.array(constraint.expression.factors(model) + [constraint.bound])
            objective_row = objective_row - factors_row

        table = np.array([objective_row] + [c.expression.factors(model) + [c.bound] for c in model.constraints])
        return t.Tableaux(model, table)

    def _basic_initial_tableaux(self, model):
        objective_row = np.array((-1 * model.objective.expression).factors(model) + [0.0])
        table = np.array([objective_row] + [c.expression.factors(model) + [c.bound] for c in model.constraints])
        return t.Tableaux(model, table)

    def _artifical_variables_are_positive(self, tableaux):
        assignment = tableaux.extract_assignment()
        for artificial_var in self.artificial_variables:
            if assignment[artificial_var.index] > 0:
                return True 
        return False

    def _remove_artificial_variables(self, tableaux):
        columns_to_remove = [var.index for var in self.artificial_variables.keys()]
        table = np.delete(tableaux.table, columns_to_remove, 1)
        return t.Tableaux(tableaux.model, table)

    def _restore_original_objective_row(self, tableaux, model):
        objective_row = np.array((-1 * model.objective.expression).factors(model) + [0.0])
        new_table = np.array(tableaux.table)
        new_table[0] = objective_row
        return t.Tableaux(model, new_table)

    def _fix_objective_row_to_the_basis(self, tableaux, basis):
        objective_row = tableaux.table[0].copy()
        for (constr_index, col) in enumerate(basis):
            row = constr_index + 1
            objective_factor = objective_row[col]
            if objective_factor < 0.0:
                objective_row = objective_row + objective_factor * tableaux.table[row]
            elif objective_factor > 0.0:
                objective_row = objective_row - objective_factor * tableaux.table[row]
        new_table = np.array(tableaux.table)
        new_table[0] = objective_row
        return t.Tableaux(tableaux.model, new_table)

    def _create_solution(self, assignment, model, initial_tableaux, tableaux, normal_model):
        assignment = [assignment[var.index] for var in model.variables]
        return s.Solution(model, assignment, initial_tableaux, tableaux, normal_model)
