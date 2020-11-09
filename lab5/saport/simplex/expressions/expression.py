from . import constraint as co

from itertools import groupby
from functools import reduce


class Expression:
    """
        A class to represent a linear polynomial in the linear programming, i.e. a sum of atom (e.g. 4x + 5y - 0.4z)

        Attributes
        ----------
        atoms : list[Atom]
            list of the atoms in the polynomial

        Methods
        -------
        __init__(*atoms : *Atom) -> Expression:
            constructs an expression with atoms given in the paremeter list
        @classmethod from_vectors(variables : Iterable[Variable], factors: Iterable[float]) -> Expression:
            constructs an expression with collections of factors and corresponding variables
        evaluate(assignment: list[float]) -> float:
            returns value of the expression for the given assignment
            assignment is just a list of values with order corresponding to the variables in the model
        simplify() -> Expression:
            returns a new expression with sorted and atoms and reduced factors 
        factors(model: Model) -> list[float]:
            return list of factors corresponding to the variables in the model
        __add__(other: Expression) -> Expression:
            returns sum of the two polynomials
        __sub__(other: Expression) -> Expression:
            returns sum of the two polynomials, inverting the first atom in the second polynomial
            useful for expressions like 3*x - 4y, otherwise one would have to write 3*x + -4*y 
        __mul__(factor: float) -> Expression:
            return a new polynomial with all factors multiplied by the given number
        __eq__(bound: float) -> Constraint:
            returns a new equality constraint
        __le__(bound: float) -> Constraint:
            returns a new "less than or equal" constraint
        __ge__(bound: float) -> Constraint:
            returns a new "greater than or equal" constraint
    """

    def __init__(self, *atoms):
        self.atoms = atoms 

    @classmethod
    def from_vectors(self, variables, factors):
        from .atom import Atom
        assert len(variables) == len(factors), f"number of factors should correspond to variables in the expression"
        atoms = [Atom(v,f) for (v,f) in zip(variables, factors)]
        return Expression(*atoms)

    def evaluate(self, assignment):
        adder = lambda val, a: val + a.evaluate(assignment[a.var.index])
        return reduce(adder, self.atoms, 0) 

    def simplify(self):
        from . import atom
        projection = lambda a: a.var.index
        reduce_atoms = lambda a1, a2: atom.Atom(a1.var, a1.factor + a2.factor)
        reduce_group = lambda g: reduce(reduce_atoms, g[1:], g[0])
    
        sorted_atoms = sorted(self.atoms, key=projection)
        grouped_atoms = [list(g[1]) for g in groupby(sorted_atoms, key=projection)]
 
        new_atoms = (reduce_group(g) for g in grouped_atoms) 
        return Expression(*new_atoms)

    def factors(self, model):
        simplified_expression = self.simplify()
        factors = [0.0 for _ in model.variables]
        for a in simplified_expression.atoms:
            factors[a.var.index] = a.factor
        return factors

    def __add__(self, other):
        new_atoms = list(self.atoms)
        new_atoms += other.atoms;
        return Expression(*new_atoms)

    def __sub__(self, other):
        return self.__add__(other._invert())

    def _invert(self):
        new_atoms = list(self.atoms)
        new_atoms[0] = new_atoms[0] * -1
        return Expression(*new_atoms)

    def __mul__(self, factor):
        new_atoms = [a * factor for a in self.atoms]
        return Expression(*new_atoms)

    __rmul__ = __mul__

    def __eq__(self, bound):
        return co.Constraint(self, bound, co.ConstraintType.EQ)
    
    def __ge__(self, bound):
        return co.Constraint(self, bound, co.ConstraintType.GE)

    def __le__(self, bound):
        return co.Constraint(self, bound, co.ConstraintType.LE)

    def __str__(self):
        text = str(self.atoms[0])
        
        for atom in self.atoms[1:]:
            text += ' + ' if atom.factor >= 0 else ' - '
            factor = "" if abs(atom.factor) == 1.0 else f"{abs(atom.factor)}*"
            text += f'{factor}{atom.var.name}'
        return text