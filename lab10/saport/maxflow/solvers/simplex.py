from .solver import AbstractSolver
from ...simplex.model import Model as LinearModel
from ...simplex.expressions.expression import Expression as LinearExpression
from ..model import Network 

class SimplexSolver(AbstractSolver):

    def solve(self) -> int:
        m = LinearModel(self.network.name)
        vars = {(u,v) : m.create_variable(f"f({u},{v})") for u,v in self.network.digraph.edges()}

        for (u,v), var in vars.items():
            m.add_constraint(var <= Network.capacity(self.network.digraph, u, v))

        for u in self.network.digraph.nodes():
            if u in { self.network.sink_node, self.network.source_node }:
                continue

            preds = [v for v in self.network.digraph.predecessors(u) if v != self.network.sink_node]
            succs = [v for v in self.network.digraph.successors(u) if v != self.network.source_node]

            inflows_vars = [vars[(v, u)] for v in preds]
            outflows_vars = [vars[(u, v)] for v in succs]

            expression = LinearExpression.from_vectors(inflows_vars + outflows_vars, [1.0] * len(inflows_vars) + [-1.0] * len(outflows_vars))
            m.add_constraint(expression == 0)

        source_vars = [vars[(u,v)] for (u,v) in vars.keys() if u == self.network.source_node]
        expression = LinearExpression.from_vectors(source_vars, [1.0] * len(source_vars))
        m.maximize(expression)

        solution = m.solve()
        return int(solution.objective_value())


