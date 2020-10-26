from copy import deepcopy
from . import model as m 
from .expressions import objective as o 
from .expressions import constraint as c
from .exceptions import Unbounded
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')




class Solution:
    """
        A class to represent a solution to linear programming problem.


        Attributes
        ----------
        model : Model
            model corresponding to the solution
        assignment : list[float]
            list with the values assigned to the variables
            order of values should correspond to the order of variables in model.variables list


        Methods
        -------
        __init__(model: Model, assignment: list[float]) -> Solution:
            constructs a new solution for the specified model and assignment
        value(var: Variable) -> float:
            returns a value assigned to the specified variable
        objective_value()
            returns value of the objective function
    """

    def __init__(self, model, assignment):
        "Assignment is just a list of values"
        self.assignment = assignment
        self.model = model

    def value(self, var):
        return self.assignment[var.index]

    def objective_value(self):
        return self.model.objective.evaluate(self.assignment)       

    def __str__(self):
        text = f'- objective value: {self.objective_value()}\n'
        text += '- assignment:'
        for (i,val) in enumerate(self.assignment):
            text += f'\n\t- {self.model.variables[i].name} = {val}'
        return text

class Tableaux:
    """
        A class to represent a solution to linear programming problem.


        Attributes
        ----------
        model : Model
            model corresponding to the tableaux
        table : numpy.Array
            2d-array with the tableaux

        Methods
        -------
        __init__(model: Model, solution: Solution) -> Tableaux:
            constructs a new tableaux for the specified model and solution
        cost_factors() -> numpy.Array:
            returns a vector containing factors in the cost row
        cost() -> float:
            returns the cost of solution represented in tableaux
        is_optimal() -> bool:
            checks whether the current solution is optimal
        choose_entering_variable() -> int:
            finds index of the variable, that should enter the basis next
        is_unbounded(col: int) -> bool:
            checks whether the problem is unbounded
        choose_leaving_variable(col: int) -> int:
            finds index of the variable, that should leave the basis next
        pivot(col: int, row: int):
            updates tableaux using pivot operation with given entering and leaving variables
        extract_solution() -> Solution:
            returns solution corresponding to the tableaux
        extract_basis() -> list[int]
            returns list of indexes corresponding to the variables belonging to the basis
    """

    def __init__(self, model):
        self.model = model
		# the "z column" is always constant so we don't include it in our table
        cost_row = np.array((-1 * model.objective.expression).factors(model) + [0.0])

        self.table = np.array([cost_row] + [c.expression.factors(model) + [c.bound] for c in model.constraints])


    def cost_factors(self):
        return self.table[0,:-1] 

    def cost(self):
        return self.table[0, -1]

    def is_optimal(self):
        # if all factors in the cost row are >= 0
        return all(factor >= 0 for factor in self.cost_factors())

    def choose_entering_variable(self):
        # return column index with the smallest factor in the cost row
        cost_factors = list(self.cost_factors())
        return cost_factors.index(min(cost_factors))


    def is_unbounded(self, col):
        # if all factors in the specified column are <= 0
        return all(factor <= 0 for factor in self.table[:, col])


    def choose_leaving_variable(self, col):
        # return row index associated with the leaving variable
        # to choose the row, divide beta column (last column) by the specified column
        # then choose a row index associated with the smallest positive value in the result
        # tip: take care to not divide by 0 :)

        logging.debug('\n-- choose_leaving_variable({0}) -------------------------------------------------'.format(col))

        # get columns but ignore first row
        specified_column = self.table[1:, col]
        RHS = self.table[1:, -1]
        logging.debug('Specified column = {0}'.format(specified_column))
        logging.debug('Last column = {0}'.format(RHS))

        # divide RHS by specified_column, if there is zero division it's fine - result will be np.inf
        # so we won't consider it later cus we're searching for smallest element
        with np.errstate(divide='ignore'):
            divided = list(map(lambda x, y: x / y, RHS, specified_column))
        logging.debug('Divided = {0}'.format(divided))

        # convert all 0 and negative numbers to np.inf so we won't consider it later
        divided_zero_ignored = [x if x > 0 else np.inf for x in divided]
        logging.debug('Divided without zeros = {0}'.format(divided_zero_ignored))

        # get index of smallest element
        # + 1 to ignore first row
        return divided_zero_ignored.index(min(divided_zero_ignored)) + 1


    def pivot(self, row, col):
        # 1) the pivot row (row at index 'row') has to be devided by the old value of the table[row, col]
        # 2) the pivot column (column at index col) now belongs to basis and should contain only 0s
        #    except the table[row,col] where it should equal 1 (already done in the first step)
        # 3) all other cells (neither in the pivot row nor column) are updated with the following rule:
        #    t'[r,c] = t[r,c] + (-t[r,col]) * t'[row, c]
        #    where:
        #       * t' = new tableaux
        #       * t = old tableaux
        #       * row, col = pivot row and columns accordingly
        #       * r, c = cell coordinates

        logging.debug('\n-- pivot({0}, {1}) --------------------------------------------------------------'.format(row, col))

        # divide chosen row by divisor
        divisor = self.table[row, col]
        for i in range(len(self.table[row, :])):
            self.table[row, i] /= divisor


        logging.debug('Divisor(pivot) table[{0}][{1}] = {2}'.format(row, col, divisor))
        logging.debug('Pivot row after division = {0}'.format(self.table[row, :]))

        # produce recalculations list of tuples to change values of particular rows later
        recalculations = [(index, -val) for index, val in enumerate(self.table[:, col]) if index != row]
        logging.debug('Recalculations = {0}'.format(recalculations))

        # changing values of particular rows
        for r, val in recalculations:
            for c in range(len(self.table[r, :])):
                self.table[r, c] += val * self.table[row, c]







    def extract_solution(self):
        # prepare assignments for corresponding variables
        assignments = [0 for _ in range(len(self.model.variables))]
        for r, c in enumerate(self.extract_basis()):
            assignments[c] = self.table[r + 1, -1]

        return Solution(self.model, assignments)


    def extract_basis(self):
        rows_n, cols_n = self.table.shape
        basis = [-1 for _ in range(rows_n - 1)]
        for c in range(cols_n - 1):
            # check every column
            column = self.table[:, c]

            # make condition for the column
            belongs_to_basis = column.min() == 0.0 and column.max() == 1.0 and column.sum() == 1.0

            if belongs_to_basis:
                # get row
                row = np.where(column == 1.0)[0][0]

                # [row-1] because we ignore the cost variable in the basis
                basis[row-1] = c
        return basis


    def __str__(self):
        def cell(x, w):
            return '{0: >{1}}'.format(x, w)

        cost_name = "z"
        basis = self.extract_basis()
        header = ["basis", cost_name] + [var.name for var in self.model.variables] + ["b"]
        longest_col = max([len(h) for h in header])

        
        rows = [[cost_name]] + [[self.model.variables[i].name] for i in basis]

        for (i,r) in enumerate(rows):
            cost_factor = 0.0 if i > 0 else 1.0
            r += [str(v) for v in [cost_factor] + list(self.table[i])]
            longest_col = max(longest_col, max([len(v) for v in r]))

        header = [cell(h, longest_col) for h in header]
        rows = [[cell(v, longest_col) for v in row] for row in rows]

        cell_sep = " | "

        result = cell_sep.join(header) + "\n"
        for row in rows:
            result += cell_sep.join(row) + "\n"
        return result


class Solver:
    """
        A class to represent a simplex solver.

        Methods
        -------
        solve(model: Model) -> Solution:
            solves the given model and return the first solution
    """

    def solve(self, model, print_tableaux, stop_on_iteration):
        normal_model = self._normalize_model(deepcopy(model))
        tableaux = Tableaux(normal_model)



        iter = 0
        while not tableaux.is_optimal():
            # additional procedure for printing tableaux
            if print_tableaux:
                logging.info('\n\n- tableaux after {0} iteration:\n{1}'.format(iter, tableaux))

            pivot_col = tableaux.choose_entering_variable()
            if tableaux.is_unbounded(pivot_col):
                raise Unbounded("Linear Programming model is unbounded")
            pivot_row = tableaux.choose_leaving_variable(pivot_col)

            tableaux.pivot(pivot_row, pivot_col)
            iter += 1

            if stop_on_iteration:
                input("- press enter to continue...")
            logging.debug('------------------------------------------------------------------------------')

        if print_tableaux:
            logging.info('\n\n- tableaux after {0} iteration:\n{1}'.format(iter, tableaux))
        
        return tableaux.extract_solution()

    def _normalize_model(self, model):
        """
            _normalize_model(model: Model) -> Model:
                returns a normalized version of the given model 
        """
        if model.objective.type == o.ObjectiveType.MIN:
            model.objective.invert()
        
        self.slack_variables = dict()
        for (i,constraint) in enumerate(model.constraints):
            if constraint.type != c.ConstraintType.EQ:
                slack_var = model.create_variable(f"s{i}")
                self.slack_variables[slack_var.index] = i
                
                if constraint.bound < 0:
                    constraint.invert()
                
                constraint.expression = constraint.expression + slack_var * c.ConstraintType.LE.value * -1
                constraint.type = c.ConstraintType.EQ 
        return model


        
        
        
        
        
        

