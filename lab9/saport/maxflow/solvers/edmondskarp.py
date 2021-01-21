from .solver import AbstractSolver
from ..model import Network

import networkx as nx
from typing import List
import time
from collections import deque


class EdmondsKarp(AbstractSolver):
    visited_attribute_key = 'visited'

    def solve(self) -> int:
        # basic body for the edmonds-karp algorithm
        max_flow = 0
        rgraph = self.create_residual_graph()
        apath = self.find_augmenting_path(
            rgraph, self.network.source_node, self.network.sink_node)
        while apath != None:
            max_flow += self.update_residual_graph(rgraph, apath)
            apath = self.find_augmenting_path(
                rgraph, self.network.source_node, self.network.sink_node)
        return max_flow

    def create_residual_graph(self) -> nx.DiGraph:
        rgraph = nx.DiGraph()
        digraph = self.network.digraph
        for start, end in digraph.edges:
            capacity = Network.capacity(digraph, start, end)
            rgraph.add_edge(start, end, capacity=capacity)
            rgraph.add_edge(end, start, capacity=0)
        return rgraph

    def find_augmenting_path(self, graph: nx.DiGraph, src: int, sink: int) -> List[int]:
        queue = deque()
        queue.append([src])
        while queue:
            path = queue.pop()
            path_end = path[-1]
            if path_end == sink:
                self._clear_visited(graph)
                return path
            for node in self._get_adjacency_list(graph, path_end):
                if not self._is_visited(graph, node):
                    self._set_visited(graph, node)
                    queue.append([*path, node])

    def _get_adjacency_list(self, graph: nx.DiGraph, node):
        return [end for start, end in graph.out_edges(node) if Network.capacity(graph, start, end) > 0]

    def _is_visited(self, graph: nx.DiGraph, node):
        return graph.nodes()[node].setdefault(EdmondsKarp.visited_attribute_key, False)

    def _set_visited(self, graph: nx.DiGraph, node):
        nx.set_node_attributes(
            graph, {node: {EdmondsKarp.visited_attribute_key: True}})

    def _clear_visited(self, graph: nx.DiGraph):
        nx.set_node_attributes(graph, False, EdmondsKarp.visited_attribute_key)

    def update_residual_graph(self, graph: nx.DiGraph, path: List[int]) -> int:
        flow = float('inf')
        # Computer Science
        edges = list(zip([path[0]] + path, path))[1:]
        # 1) find flow = min capacity of all the edges on the path
        for start, end in edges:
            capacity = Network.capacity(graph, start, end)
            if capacity < flow:
                flow = capacity
        for start, end in edges:
            # 2) for every edge in the path decrease capacity by the flow
            capacity = Network.capacity(graph, start, end)
            Network.set_capacity(graph, start, end, capacity-flow)
            # 3) for every edge opposite to the one in the path, increase its capacity by the flow
            Network.set_capacity(graph, end, start, flow)
        return flow
