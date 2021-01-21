import logging
from saport.simplex.model import Model 

def run():
    model = Model("example_05_unfeasible")

    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")

    model.add_constraint(2*x1 - x2 <= -1)
    model.add_constraint(x1 + x2 == 3)
    model.add_constraint(x1 + x2 >= 4)
    
    model.maximize(x1 + 3 * x2)

    solution = model.solve()
    assert solution.is_feasible == False, "Your algorithm found a solution to an unfeasible problem. This shouldn't happen..."
    logging.info("Congratulations! This problem is unfeasible and your algorithm has found that :)")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()
