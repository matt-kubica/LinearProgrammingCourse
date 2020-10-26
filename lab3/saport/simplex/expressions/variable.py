from . import atom as a

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