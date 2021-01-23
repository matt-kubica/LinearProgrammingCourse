from tests.knapsack.knapsack_benchmark import KnapsackBenchmark
from .knapsack_benchmark import KnapsackBenchmark

problems = ["ks_4_0", "ks_19_0", "ks_30_0", "ks_40_0", "ks_45_0", "ks_50_0", "ks_50_1", "ks_60_0", "ks_82_0", "ks_100_0", "ks_100_1", "ks_100_2", "ks_106_0", "ks_200_0", "ks_200_1", "ks_300_0", "ks_400_0", "ks_500_0"]
benchmark = KnapsackBenchmark(problems)
benchmark.run()