from . import expression as e 

class Atom(e.Expression):
    """
        A class to represent an atom of the linear programming expression, i.e. variable and it's factor (e.g. 4x, -5.3x, etc.)
        It derives from the Expression class and can be intepreted as a expression containing only single atom, itself

        Attributes
        ----------
        var : Variable
            variable associated with the atom
        factor : float
            factor value associated with the atom

        Methods
        -------
        __init__(var: Variable, factor: float) -> Atom:
            constructs new atom with a specified variable and factor
        evaluate(assigned_value: float) -> float:
            returns value of the atom for the given assignment
        __mul__(factor: float) -> Atom:
            return new atom with a multiplied factor
    """

    def __init__(self, var, factor):
        self.var = var 
        self.factor = float(factor)
        super().__init__(self)

    def evaluate(self, assigned_value):
        return self.factor * assigned_value

    def __mul__(self, factor):
        return Atom(self.var, self.factor * factor)

    __rmul__ = __mul__

    def __str__(self):
        return f"{self.factor}*{self.var}"