from saport.maxflow.model import Network 
from saport.maxflow.solvers.networkx import NetworkXSolver
from saport.maxflow.solvers.simplex import SimplexSolver
from saport.maxflow.solvers.edmondskarp import EdmondsKarp

def run_example(network_path):
    net = Network.from_file(network_path)
    baseline = NetworkXSolver(net).solve()
    simplex = SimplexSolver(net).solve()
    ek = EdmondsKarp(net).solve()
    print("- max flows calculated by various solvers:", flush = True)
    print(f"* networkx (baseline): {baseline}", flush = True)
    print(f"* simplex: {simplex}", flush = True)
    print(f"* Edmonds-Karp: {ek}", flush = True)
    
    assert simplex == baseline, "Your simplex implementation returned incorrect max flow"
    assert ek == baseline, "Your Edmonds-Karp implementation returned incorrect max flow"
