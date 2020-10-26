import logging
from saport.simplex.model import Model
from saport.simplex.exceptions import Unbounded

def run():
    # fill missing test based on the example_01_solvable.py
    # to make the test a bit more interesting:
    # * make the model unbounded!
    # 
    # TIP: you may use other solvers (e.g. https://online-optimizer.appspot.com)
    #      to check if the problem is unbounded
    model = Model("example_03"
                  "_unbounded")
    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")
    x3 = model.create_variable("x3")

    model.add_constraint(x1 - x2 - x3 >= -50)
    model.add_constraint(x1 + 2 * x2 + x3 >= -10)
    model.add_constraint(4 * x2 + x3 == 100)

    model.maximize(2 * x1 - x2 + 3 * x3)

    try:
        solution = model.solve(print_tableaux=True, stop_on_iteration=True)
        logging.info(solution)
        assert (solution.assignment == [0.0, 25.0, 0.0, 25.0, 60.0]), "Your algorithm found an incorrect solution!"
        logging.info("Congratulations! This solution seems to be alright :)")
    except Unbounded as exc:
        logging.info('{0}, exiting...'.format(exc))
    except:
        raise AssertionError("This problem has a solution and your algorithm hasn't found it!")






if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()