from knapsack_benchmark import KnapsackBenchmark

# , "ks_82_0", "ks_100_0", "ks_100_1", "ks_100_2"]
problems = ["ks_50_0", "ks_50_1", "ks_60_0"]

benchmark = KnapsackBenchmark(problems)
benchmark.run()
