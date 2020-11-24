import logging
from saport.simplex.model import Model 

def run():
    model = Model("example_01_solvable")

    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")

    model.add_constraint(x1 <= 150)
    model.add_constraint(x2 <= 250)
    model.add_constraint(2*x1 + x2 <= 500)

    model.maximize(8 * x1 + 5 * x2)

    try:
        solution = model.solve()
    except:
        raise AssertionError("This problem has a solution and your algorithm hasn't found it!")

    logging.info(solution)

    assert (solution.assignment == [125.0, 250.0]), "Your algorithm found an incorrect solution!"

    logging.info("Congratulations! This solution seems to be alright :)")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()
