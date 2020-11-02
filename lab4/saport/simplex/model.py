from . import solver as s
from .expressions import variable as v
from .expressions import objective as o

class Model:
    """
        A class to represent a linear programming problem.


        Attributes
        ----------
        name : str
            name of the problem
        variables : list[Variable]
            list with the problem variable, variable with index 'i' is always stored at the variables[i]
        constraints : list[Constraint]
            list containing problem constraints
        objective : Objective
            object representing the objective function

        Methods
        -------
        __init__(name: str) -> Model:
            constructs new model with a specified name
        create_variable(name: str) -> Variable:
            returns a new variable with a specified named, the variable is automatically indexed and added to the variables list
        add_constraint(constraint: Constraint)
            add a new constraint to the model
        maximize(expression: Expression)
            sets objective to maximize the specified Expression
        minimize(expression: Expression)
            sets objective to minimize the specified Expression
        solve() -> Solution
            solves the current model using Simplex solver and returns the result
            when called, the model should already contain at least one variable and objective
    """
    def __init__(self, name):
        self.name = name
        self.variables = []
        self.constraints = []
        self.objective = None

    def create_variable(self, name):
        for var in self.variables:
            if (var.name == name):
                raise Exception(f"There is already a variable named {name}")

        new_index = len(self.variables)
        variable = v.Variable(name, new_index)
        self.variables.append(variable)
        return variable 

    def add_constraint(self, constraint):
        self.constraints.append(constraint)
         

    def maximize(self, expression):
        self.objective = o.Objective(expression, o.ObjectiveType.MAX)
    
    def minimize(self, expression):
        self.objective = o.Objective(expression, o.ObjectiveType.MIN)

    def _simplify(self):
        self.constraints = [c.simplify() for c in self.constraints]
        self.objective = self.objective.simplify()

    def solve(self):
        if len(self.variables) == 0:
            raise Exception("Can't solve a model without any variables")

        if self.objective == None:
            raise Exception("Can't solve a model without an objective")

        self._simplify()
        solver = s.Solver()
        return solver.solve(self)

    def __str__(self):
        separator = '\n\t'
        text = f'''- name: {self.name}
- variables:{separator}{separator.join([v.name for v in self.variables])}
- constraints:{separator}{separator.join([str(c) for c in self.constraints])}
- objective:{separator}{self.objective}
'''
        return text
    

