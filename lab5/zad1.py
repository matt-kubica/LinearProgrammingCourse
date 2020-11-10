# remember to print the dual model (just print()) and the analysis results (analyser.interpret_results(solution, analysis_results))
# in case of doubt refer to examples 06 and 07

from saport.simplex.analyser import Analyser
from saport.simplex.model import Model


def run():
    primal = Model("zad1")

    ss = primal.create_variable("SS")
    s = primal.create_variable("S")
    o = primal.create_variable("O")

    primal.add_constraint(2 * ss + 2 * s + 5 * o <= 40)
    primal.add_constraint(ss + 3 * s + 2 * o <= 30)
    primal.add_constraint(3 * ss + s + 3 * o <= 30)
    primal.maximize(32 * ss + 24 * s + 48 * o)

    dual = primal.dual()

    primal_solution = primal.solve()

    print(dual)

    analyser = Analyser()
    analysis_results = analyser.analyse(primal_solution)
    analyser.interpret_results(primal_solution, analysis_results)


if __name__ == '__main__':
    run()
