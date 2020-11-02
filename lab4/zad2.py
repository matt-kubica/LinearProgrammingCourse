from saport.simplex.model import Model
import logging

def run():
    model = Model('zad2')

    p1 = model.create_variable("p1")
    p2 = model.create_variable("p2")
    p3 = model.create_variable("p3")
    p4 = model.create_variable("p4")

    model.add_constraint(0.8 * p1 + 2.4 * p2 + 0.9 * p3 + 0.4 * p4 >= 1200)
    model.add_constraint(0.6 * p1 + 0.6 * p2 + 0.3 * p3 + 0.3 * p4 >= 600)

    model.minimize(9.6 * p1 + 14.4 * p2 + 10.8 * p3 + 7.2 * p4)

    try:
        solution = model.solve()
    except:
        raise AssertionError(
            "This problem has a solution and your algorithm hasn't found it!")

    logging.info(solution)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()