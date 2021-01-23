from tests.knapsack.knapsack_benchmark import KnapsackBenchmark

problems = ["ks_lecture_dp_1", "ks_lecture_dp_2", "ks_4_0", "ks_19_0"]
benchmark = KnapsackBenchmark(problems, timelimit=float('inf'))
benchmark.run()