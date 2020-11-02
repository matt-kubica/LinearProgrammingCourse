
import logging
from saport.simplex.model import Model


def run():
    model = Model("example_05_unfeasible")

    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")
    x3 = model.create_variable("x3")

    model.maximize(x1 + x2)
    model.add_constraint(x1 + x2 + x3 == 10)
    model.add_constraint(x1 - x2 + x3 >= 100)

    try:
        solution = model.solve()
    except:
        logging.info("Congratulations! You found an unfesiable solution detectable with artificial variables :)")
    else:
        raise AssertionError("This problem has no solution but your algorithm hasn't figured it out!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()
