from saport.simplex.model import Model
import logging

def run():
    model = Model('zad1')

    x1 = model.create_variable('x1')
    x2 = model.create_variable('x2')
    x3 = model.create_variable('x3')

    model.add_constraint(x1 + x2 + x3 <= 30)
    model.add_constraint(x1 + 2 * x2 + x3 >= 10)
    model.add_constraint(2 * x2 + x3 <= 20)

    model.maximize(2 * x1 + x2 + 3 * x3)


    try:
        solution = model.solve()
    except:
        raise AssertionError(
            "This problem has a solution and your algorithm hasn't found it!")

    logging.info(solution)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()