# Lab 05 - Senitivity Analysis and Dual Problems

The goal of this is lab is to:

* fill missing code in the `saport.simplex.model.Model` and `saport.simplex.analysis_tools.ObjectiveSensitivityAnalyser` classes
* you may use new tests `example_06_dual.py` and `example_07_cost_sensitivity.py` to check correctness of your code
* fill files `zadX.py` corresponding to assignments in the `assignment.pdf` file and solve the problems!

## How To Run Tests

*Warning*: to run the tests, you should first implement the simplex algorithm. Otherwise the tests will just hang due to the `pass` methods in the `saport.simplex.solver.Tableaux`.

1. Then install required packages (`numpy` only at the moment): `pip install -r requirements.txt`
2. Then you can run each example separately as an independent program: `python example_01_solvable.py`
3. Or you can ran all the tests: `python test.py`


## SAPORT

SAPORT = Student's Attempt to Produce an Operation Research Toolkit

Refer to the `example.py` for a quick overview of the API.