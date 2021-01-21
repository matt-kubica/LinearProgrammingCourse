# Lab 09 - Max-Flow

The goal of this is lab is to implement two algorithms to find the maximum flow in a flow network. 

TODO:
* fill missing code in: 
  * `saport.maxflow.solvers.simplex`
  * `saport.maxflow.solvers.edmondskarp`
  * `networks/07_11_wikipedia.dnf`
* run `maxflow_test.py` to test your code on all available instances
* tip: `networkx` package is required, install it via `pip` (check `requirements.txt`)

## SAPORT

SAPORT = Student's Attempt to Produce an Operation Research Toolkit

Includes:

* two-step simplex
* knapsack
* integer
* min-max (2-players zero-sum games)
* max-flow

To run functional tests from the `tests` subdirectory, run commands with a correct `PYTHONPATH`, e.g.

`PYTHONPATH=. python tests/simplex/test.py`