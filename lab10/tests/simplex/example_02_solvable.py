import logging
from saport.simplex.model import Model 

def run():
    model = Model("example_02_solvable")

    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")
    x3 = model.create_variable("x3")

    model.add_constraint(x1 + 3*x2 + 2*x3 <= 10)
    model.add_constraint(-1*x1 - 5*x2 - 1*x3 >= -8)

    model.minimize(-8 * x1 - 10 * x2 - 7 * x3)

    try:
        solution = model.solve()
    except:
        raise AssertionError("This problem has a solution and your algorithm hasn't found it!")

    logging.info(solution)

    assert (solution.assignment == [8.0, 0.0, 0.0]), "Your algorithm found an incorrect solution!"

    logging.info("Congratulations! This solution seems to be alright :)")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()
