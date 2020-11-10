from copy import deepcopy
import enum
from itertools import permutations

from . import solver as s
from .expressions import expression as ex
from .expressions import variable as va
from .expressions import objective as ob
from .expressions import constraint as co
from .expressions.expression import Expression
import numpy as np
from .solver import Solver


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
        create_variable(name: str) -> Variable
            returns a new variable with a specified named, the variable is automatically indexed and added to the variables list
        add_constraint(constraint: Constraint)
            add a new constraint to the model
        maximize(expression: Expression)
            sets objective to maximize the specified Expression
        minimize(expression: Expression)
            sets objective to minimize the specified Expression
        translate_to_standard_form() -> Model
            creates a new equivalent model in a standard form (max objective and <= / = constraints)
        is_equivalent(other: Model) -> bool
            checks whether the model is equivalent to another one (ignores variables' names, etc.), useful when writing tests
        dual() -> Model
            creates a dual model 

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
        variable = va.Variable(name, new_index)
        self.variables.append(variable)
        return variable 

    def add_constraint(self, constraint):
        self.constraints.append(constraint)
         
    def maximize(self, expression):
        self.objective = ob.Objective(expression, ob.ObjectiveType.MAX)
    
    def minimize(self, expression):
        self.objective = ob.Objective(expression, ob.ObjectiveType.MIN)
        
    def _simplify(self):
        self.constraints = [c.simplify() for c in self.constraints]
        self.objective = self.objective.simplify()

    def is_equivalent(self, other):
        if not isinstance(other, Model):
            return False

        m1 = self.translate_to_standard_form()
        m2 = other.translate_to_standard_form()

        if len(m1.variables) != len(m1.variables):
            return False 
        
        if len(m1.constraints) != len(m2.constraints):
            return False 

        if m1.objective.type != m2.objective.type:
            return False

        if m1.objective.expression.factors(self) != m2.objective.expression.factors(other):
            return False 

        for (i, c) in enumerate(m1.constraints):
            oc = m2.constraints[i]
            
            if c.bound != oc.bound:
                return False 

            if c.type != oc.type:
                return False

            fs = c.expression.factors(self)
            ofs = oc.expression.factors(other)
            if fs != ofs:
                return False

        return True
        
    def dual(self):
        self._check_if_creating_dual_is_possible()
        primal = self.translate_to_standard_form()
        dual = Model(f"{primal.name} (dual)")
        self._create_dual_variables(primal, dual)
        self._create_dual_objective(primal, dual)
        self._create_dual_constraints(primal, dual)
        return dual

    def translate_to_standard_form(self):
        standard = deepcopy(self)
        standard._simplify()
        standard._change_constraints_to_LE()
        standard._change_objective_to_max()
        return standard
        
    def _check_if_creating_dual_is_possible(self):
        for constraint in self.constraints:
            if constraint.type == co.ConstraintType.EQ:
                raise Exception("Model doesn't support (yet) duals for problems with equality constraints")

    def _create_dual_variables(self, primal, dual):
        for i in range(len(primal.constraints)):
            dual.create_variable('y{0}'.format(i + 1))

    def _create_dual_objective(self, primal, dual):
        factors = [constraint.bound for constraint in primal.constraints]
        objective_expression = Expression.from_vectors(dual.variables, factors)
        dual.minimize(objective_expression)


    def _create_dual_constraints(self, primal, dual):
        # table without cost row and RHS
        primal_factors = Solver._basic_initial_tableaux(primal).table[1:, :-1]
        dual_factors = primal_factors.T

        for i, factor_row in enumerate(dual_factors):
            constraint_expression = Expression.from_vectors(dual.variables, factor_row)
            constraint = constraint_expression >= primal.objective.expression.atoms[i].factor
            dual.add_constraint(constraint)


    def _change_objective_to_max(self):
        if self.objective.type == ob.ObjectiveType.MIN:
            self.objective.invert()

    def _change_constraints_to_LE(self):
        for constraint in self.constraints:
            if constraint.type == co.ConstraintType.GE:
                constraint.invert()

    def solve(self):
        if len(self.variables) == 0:
            raise Exception("Can't solve a model without any variables")

        if self.objective == None:
            raise Exception("Can't solve a model without an objective")

        solver = s.Solver()
        return solver.solve(deepcopy(self))

    def __str__(self):
        separator = '\n\t'
        text = f'''- name: {self.name}
- variables:{separator}{separator.join([f"{v.name} >= 0" for v in self.variables])}
- constraints:{separator}{separator.join([str(c) for c in self.constraints])}
- objective:{separator}{self.objective}
'''
        return text
    

