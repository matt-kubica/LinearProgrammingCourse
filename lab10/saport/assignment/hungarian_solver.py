import numpy as np
from .model import Assignment, AssignmentProblem, NormalizedAssignmentProblem
from typing import List, Dict, Tuple, Set
from copy import deepcopy

class Solver:
    '''
    A hungarian solver for the assignment problem.

    Methods:
    --------
    __init__(problem: AssignmentProblem):
        creates a solver instance for a specific problem
    solve() -> Assignment:
        solves the given assignment problem
    extract_mins(costs: np.Array):
        substracts from columns and rows in the matrix to create 0s in the matrix
    find_max_assignment(costs: np.Array) -> Dict[int,int]:
        finds the biggest possible assinments given 0s in the cost matrix
        result is a dictionary, where index is a worker index, value is the task index
    add_zero_by_crossing_out(costs: np.Array, partial_assignment: Dict[int,int])
        creates another zero(s) in the cost matrix by crossing out lines (rows/cols) with zeros in the cost matrix,
        then substracting/adding the smallest not crossed out value
    create_assignment(raw_assignment: Dict[int, int]) -> Assignment:
        creates an assignment instance based on the given dictionary assignment
    '''
    def __init__(self, problem: AssignmentProblem):
        self.problem = NormalizedAssignmentProblem.from_problem(problem)


    def solve(self) -> Assignment:
        costs = np.array(self.problem.costs)

        while True:
            self.extracts_mins(costs)
            max_assignment = self.find_max_assignment(costs)
            if len(max_assignment) == self.problem.size():
                return self.create_assignment(max_assignment, costs)
            self.add_zero_by_crossing_out(costs, max_assignment)


    def extracts_mins(self, costs):
        height, width = costs.shape

        # rows reduction
        for row_index in range(height):
            row = costs[row_index, :]
            minimum = row.min()
            if minimum == 0: pass
            costs[row_index, :] = np.array([element - minimum for element in row])

        # columns reduction
        for col_index in range(width):
            col = costs[:, col_index]
            minimum = col.min()
            if minimum == 0: pass
            costs[:, col_index] = np.array([element - minimum for element in col])


    def add_zero_by_crossing_out(self, costs: np.array, partial_assignment: Dict[int,int]):
        markings = {
            'vertical': [],
            'horizontal': [],
        }
        crossings = deepcopy(markings)

        for row_index in range(costs.shape[0]):
            if row_index not in partial_assignment.values():
                markings['horizontal'].append(row_index)

        while True:
            i = 0
            for col_index in range(costs.shape[1]):
                column = costs[:, col_index]
                zeros_indices = np.where(column == 0)[0]
                if len(set(zeros_indices) & set(markings['horizontal'])):
                    markings['vertical'].append(col_index)
                    i += 1

            for col_index in markings['vertical']:
                if col_index in partial_assignment.keys():
                    markings['horizontal'].append(partial_assignment[col_index])

            if i > 0: break

        all_rows_indices, all_cols_indices = range(costs.shape[0]), range(costs.shape[1])
        crossings['horizontal'].extend(list(set(all_rows_indices) - set(markings['horizontal'])))
        crossings['vertical'].extend(list(set(all_cols_indices) & set(markings['vertical'])))

        # print('markings = {0}'.format(markings))
        # print('crossings = {0}'.format(crossings))

        minimum = float('inf')
        for row_index in range(costs.shape[0]):
            for col_index in range(costs.shape[1]):
                if row_index in crossings['horizontal'] or col_index in crossings['vertical']:
                    continue

                x = costs[row_index, col_index]
                if x < minimum: minimum = x

        costs -= minimum

        for row_index in crossings['horizontal']:
            costs[row_index, :] += minimum

        for col_index in crossings['vertical']:
            costs[:, col_index] += minimum


    def find_max_assignment(self, costs) -> Dict[int,int]:
        # { col_index: row_index } dictionary
        coords = {}

        while True:
            # initialize helper values
            min_zeros_amount, row_index, col_index = float('inf'), None, None

            # iterate through rows, need to find row with least '0' amount
            for row in range(costs.shape[0]):
                # if row_index already exist in coords dict, skip
                if row in coords.values():
                    continue

                # get indices of all '0' in current row, without cols already in coors dict
                zeros_indices = list(set(np.where(costs[row, :] == 0)[0]) - set(coords.keys()))

                # check amount of 0 in current row, if it's less than current minimum, proceed
                if len(zeros_indices) < min_zeros_amount:
                    i = 0
                    try:
                        # determine which '0' in the row can be used, if none of then, IndexError will be raised
                        while zeros_indices[i] in list(coords.keys()):
                            i += 1
                        col_index = zeros_indices[i]
                    except IndexError: continue

                    min_zeros_amount = len(zeros_indices)
                    row_index = row

            # if nothing could be found, one of indices would be None
            if col_index is None or row_index is None:
                break
            # otherwise append indices to coords dictionary
            coords[col_index] = row_index
        return coords


    def create_assignment(self, raw_assignment: Dict[int,int], costs) -> Assignment:
        original = self.problem.original_problem.costs
        raw_assignment = {value: key for (key, value) in raw_assignment.items()}
        assigned_tasks = [-1] * original.shape[0]

        for worker in range(original.shape[0]):
            if raw_assignment[worker] < original.shape[1]:
                assigned_tasks[worker] = raw_assignment[worker]

        total_cost = 0
        for worker, task in enumerate(assigned_tasks):
            if task < original.shape[1] and task >= 0:
                c = original[worker, task]
                total_cost += c if c >= 0 else 0

        # print('hungarian = {0}'.format(assigned_tasks))
        return Assignment(assigned_tasks, total_cost)


