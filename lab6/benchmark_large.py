from knapsack_benchmark import KnapsackBenchmark

problems = ["ks_106_0", "ks_200_0", "ks_200_1", "ks_300_0", "ks_400_0", "ks_500_0"]

benchmark = KnapsackBenchmark(problems)
benchmark.run()