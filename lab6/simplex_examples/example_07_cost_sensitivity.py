import logging
from saport.simplex.model import Model 
from saport.simplex.analyser import Analyser
from saport.simplex.analysis_tools.objective_sensitivity import ObjectiveSensitivityAnalyser
import math 

def run():
    model = Model("example_07_cost_sensitivity")

    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")
    x3 = model.create_variable("x3")

    model.add_constraint(6*x1 + 5*x2 + 8*x3 <= 60)
    model.add_constraint(10*x1 + 20*x2 + 10*x3 <= 150)
    model.add_constraint(x1 <= 8)

    model.maximize(5*x1 + 4.5*x2 + 6*x3)

    solution = model.solve()

    analyser = Analyser()
    analysis_results = analyser.analyse(solution)
    analyser.interpret_results(solution, analysis_results, logging.info)

    objective_analysis_results = analysis_results[ObjectiveSensitivityAnalyser.name()]
    expected_bounds = [(4.636, 5.4), (4.167, 6.5), (float("-inf"), 6.571)]
    tolerance = 0.001
    for (i, bounds_pair) in enumerate(objective_analysis_results):
        assert math.isclose(bounds_pair[0], expected_bounds[i][0], abs_tol=tolerance), f"left bound of the coefficient range seems to be incorrect, expected {expected_bounds[i][0]}, got {bounds_pair[0]}"
        assert math.isclose(bounds_pair[1], expected_bounds[i][1], abs_tol=tolerance), f"right bound of the coefficient range seems to be incorrect, expected {expected_bounds[i][1]}, got {bounds_pair[1]}"

    logging.info("Congratulations! This cost coefficients analysis look alright :)")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()
