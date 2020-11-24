# Lab 06 - Knapsack Problem

The goal of this is lab is to:

* fill missing code in: 
  * `saport.knapsack.solvers.greedy`
  * `saport.knapsack.solvers.greedy_density`
  * `saport.knapsack.solvers.greedy_value`
  * `saport.knaspack.solvers.greedy_weight`
  * `saport.knapsack.solvers.dynamic`
  * `saport.knaspack.solvers.bnb`
  * `saport.knapsack.solvers.bnb_dfs`
* the `saport.knaspack.solvers.dfs` is already implemented, you may use the `benchmarks_small.py` to compare your implementations with it
* run all the benchmarks `benchmark_<size>.py` and save the results into corresponding `benchark_<size>_results.txt` (update: `large` benchmark won't be graded due to its high memory requirements)

## SAPORT

SAPORT = Student's Attempt to Produce an Operation Research Toolkit

Includes:

* two-step simplex
* knapsack solver