from minimax_example import run_example
from saport.minimax.model import Equilibrium, Strategy

eq = Equilibrium(1.0, Strategy([0,1,0,0]), Strategy([0,0,1,0,0]))

run_example("tests/minimax/games/4_5_pure.txt", [eq], eq)