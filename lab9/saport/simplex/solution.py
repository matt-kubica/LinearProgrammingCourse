class Solution:
    """
        A class to represent a solution to linear programming problem.


        Attributes
        ----------
        model : Model
            model corresponding to the solution
        assignment : list[float] | None
            list with the values assigned to the variables if solution is feasible and bounded, otherwise None
            order of values should correspond to the order of variables in model.variables list
        initial_tableaux: Tableaux
            a simplex tableaux corresponding to the first base solution
        tableaux: Tableaux
            a simplex tableaux corresponding to the solution 
        normal_model: Model
            normal model with slack and surplus variables
        is_feasible: bool
            whether the problem is feasible
        is_bounded: bool
            whether the problem is bounded


        Methods
        -------
        __init__(model: Model, assignment: list[float] | None, initial_tableaux: Tableaux, tableaux: Tableaux, normal_model: Model,  is_feasible: bool, is_bounded: bool) -> Solution:
            constructs a new solution for the specified model, assignment, tableaux and normal model
            if the assignment is null, one of the flags should false - either the solution is infeasible or is unbounded
        value(var: Variable) -> float | None:
            returns a value assigned to the specified variable if the model is feasible and bounded, otherwise None
        objective_value() -> float | None:
            returns a value of the objective function if the model is feasible and bounded, otherwise None
        has_assignment() -> bool:
            helper method returning info if the model is feasible and bounded, only then there is an assignment available
    """

    def __init__(self, model, assignment, initial_tableaux, tableaux, normal_model, is_feasible, is_bounded):
        self.model = model 
        self.normal_model = normal_model
        self.is_feasible = is_feasible
        self.is_bounded = is_bounded
        self.assignment = assignment
        self.tableaux = tableaux
        self.initial_tableaux = initial_tableaux

    def value(self, var):
        return None if self.assignment == None else self.assignment[var.index]

    def objective_value(self):
        return None if self.assignment == None else self.model.objective.evaluate(self.assignment) 

    def has_assignment(self):
        return self.assignment == None

    @staticmethod
    def with_assignment(model, assignment, initial_tableaux, tableaux, normal_model):
        return Solution(model, assignment, initial_tableaux, tableaux, normal_model, True, True)  

    @staticmethod
    def unfeasible(model, initial_tableaux, tableaux, normal_model):
        return Solution(model, None, initial_tableaux, tableaux, normal_model, False, True)   

    @staticmethod
    def unbounded(model, initial_tableaux, tableaux, normal_model):
        return Solution(model, None, initial_tableaux, tableaux, normal_model, True, False)

    def __str__(self):

        if not self.is_bounded:
            return "There is no optimal solution, the model is unbounded"
        
        if not self.is_feasible:
            return "There is no solution, the model is unfeasible"

        print(self.model.objective)
        text = f'- objective value: {self.objective_value()}\n'
        text += '- assignment:'
        for (i,val) in enumerate(self.assignment):
            text += f'\n\t- {self.model.variables[i].name} = {"{:.3f}".format(val)}'
        return text