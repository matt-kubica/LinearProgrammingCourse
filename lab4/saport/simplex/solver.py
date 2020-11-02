from copy import deepcopy

from . import model as m 
from .expressions import objective as o 
from .expressions import constraint as c
from .expressions import variable as v
from . import solution as s 
from . import tableaux as t
import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

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

        self._optimize(tableaux)
        solution = tableaux.extract_solution()
        return self._translate_to_original_model(solution, model)

    def _basic_initial_tableaux(self, model):
        cost_row = np.array((-1 * model.objective.expression).factors(model) + [0.0])
        table = np.array([cost_row] + [c.expression.factors(model) + [c.bound] for c in model.constraints])

        return t.Tableaux(model, table)

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
        tableaux = self._restore_original_cost_row(tableaux, model)
        tableaux = self._fix_cost_row_to_the_basis(tableaux, basis)
        return tableaux

    def _normalize_model(self, original_model):
        """
            _normalize_model(model: Model) -> Model:
                returns a normalized version of the given model 
        """

        model = deepcopy(original_model)
        self._change_objective_to_max(model)
        self._change_constraints_bounds_to_nonnegative(model)
        self.slack_variables = self._add_slack_variables(model)
        self.surplus_variables = self._add_surplus_variables(model)   
        return model

    def _create_presolve_model(self, normalized_model):
        presolve_model = deepcopy(normalized_model)
        self.artificial_variables = self._add_artificial_variables(presolve_model)
        return presolve_model    

    def _change_objective_to_max(self, model):
        if model.objective.type == o.ObjectiveType.MIN:
            model.objective.invert()

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
        return slack_variables

    def _add_surplus_variables(self, model):
        # TODO: check it later
        surplus_variables = dict()
        for (i, constraint) in enumerate(model.constraints.copy()):
            if constraint.type == c.ConstraintType.GE:
                surplus_var = model.create_variable(f"s{i}")
                surplus_variables[surplus_var] = i
                constraint.expression = constraint.expression - surplus_var
        return surplus_variables

    def _add_artificial_variables(self, model):
        artificial_variables = dict()
        for (i, constraint) in enumerate(model.constraints.copy()):
            if constraint.type in (c.ConstraintType.GE, c.ConstraintType.EQ):
                artificial_var = model.create_variable(f"R{i}")
                artificial_variables[artificial_var] = i
                constraint.expression = constraint.expression + artificial_var
        return artificial_variables


    def _first_phase_objective(self, model):
        new_model = deepcopy(model)

        # assign 0 to all present variables' factors
        self.initial_objective = model.objective
        for atom in new_model.objective.expression.atoms:
            atom.factor = 0.0

        # subtract artificial variables to objective expression
        for var in self.artificial_variables.keys():
            new_model.objective.expression -= var

        return new_model


    def _fix_basis(self, tableaux):

        cost_row = tableaux.table[0, :]
        constraints_rows = tableaux.table[1:, :]
        logging.debug('Tableaux before fixing basis:\n{0}'.format(tableaux))

        # TODO: may be to simple cus there is multiplication by 1 and substraction, maybe multiplication based on art. var factor?
        #
        for row in constraints_rows:
            for i, factor in enumerate(row):
                cost_row[i] -= factor

        logging.debug('Tableaux after fixing basis:\n{0}'.format(tableaux))
        return tableaux


    def _presolve_initial_tableaux(self, model):
        # TODO: create an initial tableaux for the artificial variables
        # - cost row should contain 1.0 for every artificial variable
        # - then you should subtract from it rows corresponding to the artificial variables
        # you may look at the _basic_initial_tableaux on how to create a tableaux

        # model with artificial variables in objective
        model = self._first_phase_objective(model)
        # tableaux with new objective row and fixed basis
        tableaux = self._fix_basis(self._basic_initial_tableaux(model))
        return tableaux

    def _artifical_variables_are_positive(self, tableaux):
        # check if indexes of basis are intercept with indexes of artificial variables
        return any(set([var.index for var in self.artificial_variables.keys()]) & set(tableaux.extract_basis()))


    def _remove_artificial_variables(self, tableaux):
        # deleting columns from tableaux according to artificial variable indices
        tableaux.table = np.delete(tableaux.table, [var.index for var in self.artificial_variables.keys()], 1)
        # deleting artificial variables from model
        [tableaux.model.variables.remove(var) for var in self.artificial_variables.keys()]
        logging.debug('Tableaux after removing artificial variables:\n{0}'.format(tableaux))
        return tableaux

    def _restore_original_cost_row(self, tableaux, initial_model):
        # change objective row to original
        tableaux.table[0, :] = self._basic_initial_tableaux(initial_model).table[0, :]
        logging.debug('Tableaux after restoring cost row:\n{0}'.format(tableaux))
        return tableaux


    def _fix_cost_row_to_the_basis(self, tableaux, basis):
        cost_row = tableaux.table[0, :]
        constraints_rows = tableaux.table[1:, :]

        for row_index, row in enumerate(constraints_rows):
            temp_cost_row = deepcopy(cost_row)
            for col_index, factor in enumerate(row):
                cost_row[col_index] = cost_row[col_index] - factor * temp_cost_row[basis[row_index]]

        tableaux.table[0, :] = cost_row
        logging.debug('Tableaux after fixing cost row to the basis:\n{0}'.format(tableaux))

        return tableaux

    def _translate_to_original_model(self, solution, model):
        assignment = [solution.value(var) for var in model.variables]
        return s.Solution(model, assignment)
