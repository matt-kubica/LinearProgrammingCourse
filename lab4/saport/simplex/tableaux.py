import numpy as np
from . import solution as s

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
        __init__(model: Model, table: array) -> Tableaux:
            constructs a new tableaux for the specified model and initial table
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

    def __init__(self, model, table):
        self.model = model
        self.table = table

    def cost_factors(self):
        return self.table[0,:-1] 

    def cost(self):
        return self.table[0, -1]

    def is_optimal(self):
        return self.cost_factors().min() >= 0

    def choose_entering_variable(self):
        return self.cost_factors().argmin()

    def is_unbounded(self, col):
        return self.table[1:, col].max() <= 0 

    def choose_leaving_variable(self, col):
        column = np.copy(self.table[1:, col])
        column = np.where(column > 0, column, -1)
        indicators = self.table[1:, -1] / column
        return np.where(indicators > 0, indicators, np.inf).argmin() + 1

    def pivot(self, row, col):
        rows_n, cols_n = self.table.shape
        pivot_factor = self.table[row, col]

        new_table = self.table.copy()
        new_table[row] = self.table[row] / pivot_factor

        new_table[:, col] = 0.0
        new_table[row, col] = 1.0
        
        for r in range(rows_n):
            if r == row:
                continue
            for c in range(cols_n):
                if c == col:
                    continue 
                new_table[r,c] = (-self.table[r, col]) * new_table[row, c] + self.table[r,c]

        self.table = new_table

    def extract_solution(self):
        rows_n, cols_n = self.table.shape
        assignment = [0.0 for _ in range(cols_n - 1)]
        basis = self.extract_basis()
        for r in range(1, rows_n):
            var_index = basis[r - 1]
            assignment[var_index] = self.table[r, -1]
        
        return s.Solution(self.model, assignment)
    
    def extract_basis(self):
        rows_n, cols_n = self.table.shape
        basis = [-1 for _ in range(rows_n -1)]
        for c in range(cols_n - 1):
            column = self.table[:,c]
            belongs_to_basis = column.min() == 0.0 and column.max() == 1.0 and column.sum() == 1.0
            if belongs_to_basis:
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
            r += ["{:.3f}".format(v) for v in [cost_factor] + list(self.table[i])]
            longest_col = max(longest_col, max([len(v) for v in r]))

        header = [cell(h, longest_col) for h in header]
        rows = [[cell(v, longest_col) for v in row] for row in rows]

        cell_sep = " | "

        result = cell_sep.join(header) + "\n"
        for row in rows:
            result += cell_sep.join(row) + "\n"
        return result