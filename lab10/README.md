# Lab 10 - Assignment Problem

The goal of this is lab is to implement two algorithms to find the best assignment of tasks to workers. 

TODO:
* fill missing code in: 
  * `saport.assignment.simplex_solver.py`
  * `saport.assignment.hungarian_solver.py`
  * `saport.assignment.model.py`
* run `test.py` to test your code on all available instances
* if one knows how to use MiniZinc, they can try to use `minizinc_test.mzn` model to check their results. Otherwise one should ignore all `*.dzn` and `mzn` files.

## SAPORT

SAPORT = Student's Attempt to Produce an Operation Research Toolkit

Includes:

* two-step simplex
* knapsack
* integer
* min-max (2-players zero-sum games)
* max-flow
* assignment problem

To run functional tests from the `tests` subdirectory, run commands with a correct `PYTHONPATH`, e.g.

`PYTHONPATH=. python tests/simplex/test.py`