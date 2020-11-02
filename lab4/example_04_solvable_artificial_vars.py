import logging
from saport.simplex.model import Model 

def run():
    model = Model("example_04_artificial_variables")

    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")

    model.add_constraint(2 * x1 - x2 <= -1)
    model.add_constraint(1 * x1 + x2 == 3)

    model.maximize(1 * x1 + 3 * x2)

    try:
        solution = model.solve()
    except:
        raise AssertionError(
            "This problem has a solution and your algorithm hasn't found it!")

    logging.info(solution)

    assert (solution.assignment == [0.0, 3.0]
            ), "Your algorithm found an incorrect solution!"

    logging.info("Congratulations! This solution seems to be alright :)")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()
