from saport.simplex.model import Model
import logging

def run():
    model = Model('zad3')

    s = model.create_variable("s")
    p = model.create_variable("p")

    model.add_constraint(15 * s + 2 * p <= 60)
    model.add_constraint(5 * s + 15 * p >= 50)
    model.add_constraint(20 * s + 5 * p >= 40)

    model.minimize(8 * s + 4 * p)

    try:
        solution = model.solve()
    except:
        raise AssertionError(
            "This problem has a solution and your algorithm hasn't found it!")

    logging.info(solution)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()