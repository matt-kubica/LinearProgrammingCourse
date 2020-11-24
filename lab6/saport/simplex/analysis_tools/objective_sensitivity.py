class ObjectiveSensitivityAnalyser:
    """
        A class used to analyse sensitivity to changes of the cost factors.


        Attributes
        ----------
        name : str
            unique name of the analysis tool

        Methods
        -------
        analyse(solution: Solution) -> List[(float, float)]
            analyses the solution and returns list of tuples containing acceptable bounds for every objective coefficient, i.e.
            if the results contain tuple (-inf, 5.0) at index 1, it means that objective coefficient at index 1 should have value >= -inf and <= 5.0
            to keep the current solution an optimum

         interpret_results(solution: Solution, results : List(float, float), print_function : Callable = print):
            prints an interpretation of the given analysis results via given print function
    """    

    @classmethod
    def name(self):
        return "Cost Coefficient Sensitivity Analysis"

    def __init__(self):
        self.name = ObjectiveSensitivityAnalyser.name()
    
    def analyse(self, solution):
        obj_coeffs = solution.normal_model.objective.expression.factors(solution.model)
        final_obj_coeffs = solution.tableaux.table[0,:-1]
        obj_coeffs_ranges = []

        basis = solution.tableaux.extract_basis()
        for (i, obj_coeff) in enumerate(obj_coeffs):
            left_side, right_side = None, None
            if i in basis:
                row = basis.index(i)
                row_coeffs = solution.tableaux.table[row + 1, :-1]
                
                left_side_bounds = [final_obj_coeffs[j] / a for (j, a) in enumerate(row_coeffs) if a > 0 and j != i]
                left_side = float('-inf') if len(left_side_bounds) == 0 else obj_coeff - min(left_side_bounds)
                right_side_bounds = [final_obj_coeffs[j] / a for (j, a) in enumerate(row_coeffs) if a < 0 and j != i]
                right_side = float('inf') if len(right_side_bounds) == 0 else obj_coeff - max(right_side_bounds)           
            else:
                left_side = float('-inf')
                right_side = obj_coeff + solution.tableaux.table[0,i]

            obj_coeffs_ranges.append((left_side, right_side))
        
        return obj_coeffs_ranges


    def interpret_results(self, solution, obj_coeffs_ranges, print_function):        
        org_coeffs = solution.normal_model.objective.expression.factors(solution.model)

        print_function("* Cost Coefficients Sensitivity Analysis:")
        print_function("-> To keep the the current optimum, the cost coefficients should stay in following ranges:")
        col_width = max([max(len(f'{r[0]:.3f}'), len(f'{r[1]:.3f}')) for r in obj_coeffs_ranges])
        for (i, r) in enumerate(obj_coeffs_ranges):
            print_function(f"\t {r[0]:{col_width}.3f} <= c{i} <= {r[1]:{col_width}.3f}, (originally: {org_coeffs[i]:.3f})")


        
    

