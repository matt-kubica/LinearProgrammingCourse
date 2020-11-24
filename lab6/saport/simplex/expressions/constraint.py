from enum import Enum

class ConstraintType(Enum):
    """
        An enum to represent a constraint type:
        - LE = less than or equal
        - EQ = equality
        - GR = greater than or equal
    """
    LE = -1
    EQ = 0
    GE = 1

    def __str__(self):
        return {
            ConstraintType.LE: "<=",
            ConstraintType.EQ: "=",
            ConstraintType.GE: ">="
        }[self]

class Constraint:
    """
        A class to represent a constraint in the linear programming expression, e.g. 4x + 5y <= 13, etc.

        Attributes
        ----------
        expression : Expression
            polynomial expressions that is bounded
        bound : float
            a bound constraining the linear polynomial
        type: ConstraintType
            type of the constraint: LE, EQ, GE

        Methods
        -------
        __init__(expression: Expression, bound: float, type: ConstraintType = ConstraintType.GE) -> Constraint:
            constructs new constraint with a specified polynomial, bound and type
        simplify() -> Constraint:
            returns new constraint with the simplified polynomial
        invert():
            inverts type of the constraint (multiplies constraint times -1)
    """
    def __init__(self, expression, bound, type = ConstraintType.GE):
        self.expression = expression
        self.bound = bound
        self.type = type

    def simplify(self):
        return Constraint(self.expression.simplify(), self.bound, self.type)

    def invert(self):
        self.type = ConstraintType(self.type.value * -1)
        self.expression = self.expression * -1
        self.bound = self.bound * -1

    def __str__(self):
        return f"{self.expression} {self.type} {self.bound}"