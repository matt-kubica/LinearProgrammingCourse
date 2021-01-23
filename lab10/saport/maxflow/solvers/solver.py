from ..model import Network

class AbstractSolver:

    def __init__(self, network: Network):
        self.network = network

    def solve(self) -> int:
        raise Exception("This is an abstract solver, don't call it directly!")