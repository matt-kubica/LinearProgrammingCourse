import sys 
from saport.assignment.model import AssignmentProblem
import numpy as np

if __name__ == "__main__":
    path = sys.argv[1]
    model = AssignmentProblem.from_file(path + ".txt")
    with open(path + ".dzn", 'w') as dzn:
        dzn.write(f"min = {'true' if model.objective_is_min else 'false'};\n")
        flat_costs = list(model.costs.flatten())
        dzn.write(f"costs = array2d(1..{model.n_workers()}, 1..{model.n_tasks()}, {flat_costs});\n")
