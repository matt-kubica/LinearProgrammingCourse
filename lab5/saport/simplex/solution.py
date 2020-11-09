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
        initial_tableaux: Tableaux
            a simplex tableaux corresponding to the first base solution
        tableaux: Tableaux
            a simplex tableaux corresponding to the solution 
        normal_model: Model
            normal model with slack and surplus variables


        Methods
        -------
        __init__(model: Model, assignment: list[float], initial_tableaux: Tableaux, tableaux: Tableaux, normal_model: Model) -> Solution:
            constructs a new solution for the specified model, assignment, tableaux and normal model
        value(var: Variable) -> float:
            returns a value assigned to the specified variable
        objective_value()
            returns value of the objective function
    """

    def __init__(self, model, assignment, initial_tableaux, tableaux, normal_model):
        "Assignment is just a list of values"
        self.assignment = assignment
        self.model = model
        self.tableaux = tableaux
        self.initial_tableaux = initial_tableaux
        self.normal_model = normal_model

    def value(self, var):
        return self.assignment[var.index]

    def objective_value(self):
        return self.model.objective.evaluate(self.assignment)       

    def __str__(self):
        text = f'- objective value: {self.objective_value()}\n'
        text += '- assignment:'
        for (i,val) in enumerate(self.assignment):
            text += f'\n\t- {self.model.variables[i].name} = {"{:.3f}".format(val)}'
        return text