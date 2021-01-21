from copy import deepcopy
from ..simplex import model as lpmodel
from . import solver as s

class Model(lpmodel.Model):
    """
        An integer programming model.
        It has the same the same structure as the linear programming model.
        
        Attributes:
        ----------
        solver: Solver
        solver used to solve the model. Useful when one wants to check some statistics, solving time, etc. 
    """

    def __str__(self):
        separator = '\n\t'
        text = f'''- name: {self.name}
- variables:{separator}{separator.join([f"{v.name} ∈ ℕ" for v in self.variables])}
- constraints:{separator}{separator.join([str(c) for c in self.constraints])}
- objective:{separator}{self.objective}
'''
        return text

    def solve(self, timelimit = float('inf')):
        if len(self.variables) == 0:
            raise Exception("Can't solve a model without any variables")

        if self.objective == None:
            raise Exception("Can't solve a model without an objective")

        self.solver = s.Solver()
        return self.solver.solve(self.translate_to_standard_form(), timelimit)