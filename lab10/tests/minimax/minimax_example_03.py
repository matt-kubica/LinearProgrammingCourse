from minimax_example import run_example
from saport.minimax.model import Equilibrium, Strategy

run_example("tests/minimax/games/3_3_mixed_neg.txt", 
            [], 
            Equilibrium(-0.14, 
                Strategy([0.43, 0.21, 0.36]),
                Strategy([0.43, 0.29, 0.29])
            ))