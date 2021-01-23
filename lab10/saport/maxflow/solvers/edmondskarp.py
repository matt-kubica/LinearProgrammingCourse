from .solver import AbstractSolver
from ..model import Network

import networkx as nx
from typing import List
import time 

class EdmondsKarp(AbstractSolver):

    def solve(self) -> int:
        max_flow = 0
        rgraph = self.create_residual_graph()
        apath = self.find_augmenting_path(rgraph, self.network.source_node, self.network.sink_node)
        while apath != None:
            max_flow += self.update_residual_graph(rgraph, apath)
            apath = self.find_augmenting_path(rgraph, self.network.source_node, self.network.sink_node)
        return max_flow
        
    def create_residual_graph(self) -> nx.DiGraph:
        rgraph = nx.DiGraph()
        rgraph.add_edges_from(self.network.digraph.edges(data = True))
        rgraph.add_edges_from([(v,u, {'capacity' : 0}) for (u,v) in self.network.digraph.edges()])
        return rgraph

    def find_augmenting_path(self, graph: nx.DiGraph, src: int, sink: int) -> List[int]:
        def successors(visited) -> List[int]:
            u = visited[-1]
            return [v for v in graph.successors(u) if v not in visited and Network.capacity(graph, u, v) > 0] 

        queue = [[src]]

        while len(queue) > 0:
            path = queue.pop(0)
            if path[-1] == sink:
                return path
            new = [path + [s] for s in successors(path)] 
            queue = queue + new
            
        return None

    def update_residual_graph(self, graph: nx.DiGraph, path: List[int]) -> int:
        min_flow = float('inf')
        path_edges = list(zip(path, path[1:]))
        for u, v in path_edges: 
            min_flow = min(min_flow, Network.capacity(graph, u, v))

        for u, v in path_edges:
            Network.set_capacity(graph, u, v, Network.capacity(graph, u, v) - min_flow)
            Network.set_capacity(graph, v, u, Network.capacity(graph, v, u) + min_flow)

        return min_flow


