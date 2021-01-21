from tests.knapsack.knapsack_benchmark import KnapsackBenchmark

problems = ["ks_50_0", "ks_50_1", "ks_60_0", "ks_100_0", "ks_100_2", "ks_200_0", "ks_500_0"]

benchmark = KnapsackBenchmark(problems)
benchmark.run()