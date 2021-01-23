from tests.knapsack.knapsack_benchmark import KnapsackBenchmark

problems = ["ks_4_0", "ks_19_0", "ks_30_0", "ks_40_0", "ks_45_0"]
benchmark = KnapsackBenchmark(problems)
benchmark.run()