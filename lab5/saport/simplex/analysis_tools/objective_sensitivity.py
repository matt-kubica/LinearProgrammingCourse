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
        #TODO: 
        # for each objective coefficient in the problem find the bounds within
        # the current optimal solution stays optimal
        #
        # tip1: obj_coeffs contains the original coefficients in the normal representation of the model
        # tip2: final_obj_coeffs is the objective row of the final tableaux, will be useful
        # tip3: obj_coeffs_ranges should contain at the end of this method pairs of bounds (left bound and right bound) for each coefficient
        # tip4: float('-inf') / float('inf') represent infinite numbers

        obj_coeffs = solution.normal_model.objective.expression.factors(solution.model)
        final_obj_coeffs = solution.tableaux.table[0,:-1]
        obj_coeffs_ranges = []

        basis = solution.tableaux.extract_basis()
        for (i, obj_coeff) in enumerate(obj_coeffs):
            left_side, right_side = None, None
            if i in basis:
                #TODO: calculate left_side and right_side for the coefficients corresponding to the variable in optimal basis
                raise Exception("not implemented yet!")
            else:
                #TODO: calculate left_side and right_side for the coefficients corresponding to the variable absent from the optimal basis
                raise Exception("not implemented yet!")

            obj_coeffs_ranges.append((left_side, right_side))
        
        return obj_coeffs_ranges


    def interpret_results(self, solution, obj_coeffs_ranges, print_function = print):        
        org_coeffs = solution.normal_model.objective.expression.factors(solution.model)

        print_function("* Cost Coefficients Sensitivity Analysis:")
        print_function("-> To keep the the current optimum, the cost coefficients should stay in following ranges:")
        col_width = max([max(len(f'{r[0]:.3f}'), len(f'{r[1]:.3f}')) for r in obj_coeffs_ranges])
        for (i, r) in enumerate(obj_coeffs_ranges):
            print_function(f"\t {r[0]:{col_width}.3f} <= c{i} <= {r[1]:{col_width}.3f}, (originally: {org_coeffs[i]:.3f})")


        
    

