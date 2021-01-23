from __future__ import annotations
import networkx as nx
from dataclasses import dataclass
from networkx.drawing.nx_agraph import to_agraph
import os.path

@dataclass
class Network:
    """
    A dataclass representing a flow network

    Attributes
    ----------
    digraph : networkx.DiGraph
        a directed graph with 'capacity' attribute on each edge
    source_node : int
        index of the source node in the network
    sink_node : int
        index of the sink node in the network
    name:
        name of the network (for debugging purposes)

    Static Methods
    --------------
    capacity(graph : networkx.DiGraph, edge_start: int, edge_end: int) -> int:
        helper method to get capacity of the given edge
    set_capacity(graph : networkx.DiGraph, edge_start: int, edge_end: int, capacity: int) -> None:
        helper method to set capacity of the given edge
    from_file(path: str) -> Network:
        creates a network based on the DNF file located at the given path
        
    Methods
    -------
    serialize(path: str) -> None:
        serializes network to a png file
    """

    digraph: nx.DiGraph
    source_node: int
    sink_node: int
    name: str

    def capacity(self, edge_start: int, edge_end: int) -> int:
        return Network.capacity(self.digraph, edge_start, edge_end)

    @staticmethod
    def capacity(graph : nx.DiGraph, edge_start: int, edge_end: int) -> int:
        return graph.get_edge_data(edge_start, edge_end)['capacity']

    @staticmethod
    def set_capacity(graph : nx.DiGraph, edge_start: int, edge_end: int, capacity: int) -> None:
        nx.set_edge_attributes(graph, {(edge_start, edge_end): {'capacity': capacity}})

    @staticmethod
    def from_file(path: str) -> Network:
        name = os.path.splitext(os.path.basename(path))[0]
        node_symb = 'n'
        start_symb = 's'
        terminal_symb = 't'
        arc_symb = 'a'

        digraph = nx.DiGraph()
        source_node = None 
        sink_node = None
        
        with open(path) as f:    
            for line in f:
                line_components = line.split() 
                if line_components[0] == arc_symb:
                    start = int(line_components[1])-1
                    end = int(line_components[2])-1
                    capacity = int(line_components[3])
                    digraph.add_edge(start, end, capacity=capacity, label=str(capacity))
                elif line_components[0] == node_symb:
                    if line_components[2] == start_symb:
                        source_node = int(line_components[1]) - 1
                    elif line_components[2] == terminal_symb:
                        sink_node = int(line_components[1]) - 1
        
        if source_node == None:
            raise Exception('Network definition is missing a source node')

        if sink_node == None:
            raise Exception('Network definition is missing a sink node')

        nx.set_node_attributes(digraph, {source_node: {'color' : 'red'}, sink_node : {'color' : 'blue'}})

        return Network(digraph, source_node, sink_node, name)  

    def serialize(self, path = None) -> None:
        output_path = f'{self.name}.png' if path is None else path
        A = to_agraph(self.digraph)
        A.layout('dot')
        A.draw(output_path)
        
    
