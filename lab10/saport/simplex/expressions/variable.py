from . import atom as a
from enum import Enum

class Variable(a.Atom):
    """
        A class to represent a linear programming variable.
        It derives from the Atom class and can be interpreted as Atom with factor = 1.

        Attributes
        ----------
        name : str
            name of the variable
        index : int
            index of the variable used in the model
        type : VariableType
            type of the variable

        Methods
        -------
        __init__(name: str, index: int) -> Variable:
            constructs new variable with a specified name and index
    """
    def __init__(self, name, index):
        self.name = name
        self.index = index
        super().__init__(self, 1)

    def __str__(self):
        return self.name

    def __key(self):
        return (self.name, self.index)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.__key() == other.__key()
        return NotImplemented