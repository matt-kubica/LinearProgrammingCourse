from saport.minimax.model import Game 
from saport.minimax.solvers.pure import PureSolver 
from saport.minimax.solvers.mixed import MixedSolver 

def run_example(game_path, expected_pure, expected_mixed):
    game = Game.from_file(game_path)
    print(f"- original game: {game}")
    print("----------")

    pure_solver = PureSolver(game)
    pure_eqs = pure_solver.solve()
    print("- pure equilibriums:")
    if len(pure_eqs) == 0:
        print("There is no pure equilibrium in this game")
    else:
        print("===".join([f"{eq}" for eq in pure_eqs]))
    print("----------")

    mixed_solver = MixedSolver(game)
    mixed_eq = mixed_solver.solve() 
    print("- mixed equilibrium:")
    print(mixed_eq)
    print("----------")
    assert len(pure_eqs) == len(expected_pure), "Your algorithm found incorrect amount of pure equilibriums!"
    for i,eq in enumerate(pure_eqs):
        assert eq.equals_enough(expected_pure[i])
    assert mixed_eq.equals_enough(expected_mixed), "Your algorithm found an incorrect pure equilibrium!"

    print("Congratulations! Your algorithm found correct Equilibriums")