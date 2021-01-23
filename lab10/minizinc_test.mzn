include "globals.mzn";

% Use this editor as a MiniZinc scratch book
array[int,int] of int: costs;

bool: min;

set of int: Workers = index_set_1of2(costs);
set of int: Tasks = index_set_2of2(costs);
set of int: TasksWithDummy = 0..n_tasks;
int: n_workers = length(Workers);
int: n_tasks = length(Tasks);
int: n_assignable_tasks = min(n_workers, n_tasks);

array[Workers] of var TasksWithDummy: assignment;

array[Workers,TasksWithDummy] of int: costs_with_dummy = array2d(Workers, TasksWithDummy, [
  if (t == 0) then 0 else costs[w,t] endif
| w in Workers, t in TasksWithDummy]);

constraint trace("\(n_assignable_tasks)", true);

constraint alldifferent_except_0(assignment);
constraint count(assignment, 0) = n_workers - n_assignable_tasks;

var int: total_cost = sum([costs_with_dummy[w, assignment[w]] | w in Workers]);
var int: objective = if (min) then -total_cost else total_cost endif;
solve maximize objective;

output ["cost: \(total_cost)\n"] ++ ["assignment: \([assignment[w] + 1 | w in Workers])"];