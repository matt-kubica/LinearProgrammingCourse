from .analysis_tools.objective_sensitivity import ObjectiveSensitivityAnalyser

class Analyser:
    """
        A class to run analysis based on the simplex solution

        Attributes
        ----------
        tools : List
            list of tools used for the analysis


        Methods
        -------
        analyse(solution: Solution) -> List:
            returns list of various analysis results for the given solution
        interpret_results(solution: Solution, results : List, print_function : Callable = print):
            prints an interpretation of the given analysis results via given print function
    """
    
    def __init__(self):
        self.tools = [ObjectiveSensitivityAnalyser()]

    def analyse(self, solution):
        result = dict()
        for tool in self.tools:
            result[tool.name] = tool.analyse(solution)
        return result

    def interpret_results(self, solution, results, print_function = print):
        for tool in self.tools:
            tool.interpret_results(solution, results[tool.name], print_function)