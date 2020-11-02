from saport.simplex.model import Model
import logging

def run():
    model = Model('zad4')

    xs = [model.create_variable(f'x{i}') for i in range(14)]
    # constraints
    model.add_constraint(xs[0] + xs[1] + xs[2] + xs[3] ==
                         150)  # all cuts, that contain type 1
    model.add_constraint(xs[1] + xs[4] + xs[5] + xs[6] + xs[7] +
                         2 * xs[8] + 2 * xs[9] == 200)  # all cuts, that contain type 2
    model.add_constraint(xs[2] + 2 * xs[3] + xs[5] + 2 * xs[6] + 3 * xs[7] + 1 * xs[9] + 2 *
                         xs[10] + 3 * xs[11] + 4 * xs[12] + 5 * xs[13] == 150)  # all cuts, that contain type 3

    model.minimize(
        95 * xs[0] + 20 * xs[1] + 60 * xs[2] + 25 * xs[3] + 125 * xs[4] + 90 * xs[5] + 55 * xs[6] + 20 * xs[7] + 50 *
        xs[8] + 15 * xs[9] + 130 * xs[10] + 95 * xs[11] + 60 * xs[12] + 25 * xs[13])

    solution = model.solve()
    logging.info(solution)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()