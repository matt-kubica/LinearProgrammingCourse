from minimax_example import run_example
from saport.minimax.model import Equilibrium, Strategy

run_example("tests/minimax/games/3_3_mixed_nonneg.txt", 
            [], 
            Equilibrium(2.0, 
                Strategy([0, 0, 1]),
                Strategy([0.8, 0.2, 0])
            ))