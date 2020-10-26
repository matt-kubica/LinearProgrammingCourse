import logging
from saport.simplex.model import Model
from saport.simplex.exceptions import Unbounded


def run():
    model = Model("example_00_very_simple")

    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")

    model.add_constraint(3 * x1 + 5 * x2 <= 78)
    model.add_constraint(4 * x1 + 1 * x2 <= 36)
    model.maximize(5 * x1 + 4 * x2)

    try:
        solution = model.solve(print_tableaux=True, stop_on_iteration=True)
        logging.info(solution)
        assert (solution.assignment == [6.0, 12.0, 0.0, 0.0]), "Your algorithm found an incorrect solution!"
        logging.info("Congratulations! This solution seems to be alright :)")
    except Unbounded as exc:
        logging.info('{0}, exiting...'.format(exc))
    except:
        raise AssertionError("This problem has a solution and your algorithm hasn't found it!")




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()
