import logging
from saport.simplex.model import Model 

def run():
    primal = Model("example_06_dual")

    x0 = primal.create_variable("x0")
    x1 = primal.create_variable("x1")
    x2 = primal.create_variable("x2")

    primal.add_constraint(4*x0 + 8*x1 - x2 <= 5)
    primal.add_constraint(7*x0 - 2*x1 + 2*x2 >= 4)
    
    primal.maximize(3*x0 + 2*x1 - 6*x2)

    expected_dual = Model("example_06_dual (expected dual)")

    y0 = expected_dual.create_variable("y0")
    y1 = expected_dual.create_variable("y1")

    expected_dual.add_constraint(4*y0 - 7*y1 >= 3)
    expected_dual.add_constraint(8*y0 + 2*y1 >= 2)
    expected_dual.add_constraint(-1*y0 - 2*y1 >= -6)

    expected_dual.minimize(5 * y0 - 4 * y1)
    
    dual = primal.dual()

    assert dual.is_equivalent(expected_dual), "dual wasn't calculated as expected"
    assert primal.is_equivalent(dual.dual()), "double dual should equal the initial model"
    
    primal_solution = primal.solve()
    dual_solution = dual.solve()

    assert primal_solution.objective_value() == dual_solution.objective_value(), "dual and primal should have the same value at optimum"
    
    logging.info("Congratulations! The dual creation seems to be implemented correctly :)")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()
