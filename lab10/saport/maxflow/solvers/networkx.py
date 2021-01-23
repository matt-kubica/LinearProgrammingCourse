from .solver import AbstractSolver
import networkx as nx
class NetworkXSolver(AbstractSolver):

    def solve(self) -> int:
        return nx.maximum_flow_value(self.network.digraph, self.network.source_node, self.network.sink_node)