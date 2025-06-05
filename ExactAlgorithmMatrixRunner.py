import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Callable
from ExactAlgorithm.ExactAlgorithmClass import ExactTSPSolver

def read_distance_matrix_from_txt(file_path: str) -> list:
    with open(file_path, 'r') as f:
        return [list(map(float, line.strip().split())) for line in f if line.strip()]

# =============================================================================
#                               Plot
# =============================================================================
def plot_running_time_vs_number_of_cities(city_nums, results_by_algorithm, algorithms):
    plt.figure(figsize=(10, 6))
    for algo_name in algorithms:
        plt.plot(city_nums, results_by_algorithm[algo_name]['times'], marker='o', label=algo_name)
    plt.xlabel("Number of Cities")
    plt.ylabel("Running Time (seconds)")
    plt.title("Running Time vs Number of Cities")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison_runtime.png')
    plt.show()


def plot_cost_comparison(filenames, results_by_algorithm, algorithms):
    bar_width = 0.2
    x = np.arange(len(filenames))
    plt.figure(figsize=(12, 6))

    for algo_name, offset in zip(algorithms, range(len(algorithms))):
        plt.bar(x + offset * bar_width, results_by_algorithm[algo_name]['costs'],
                width=bar_width, label=algo_name)

    plt.xlabel("Instance File")
    plt.ylabel("Tour Cost")
    plt.title("Tour Cost: Computed by Algorithms")
    plt.xticks(x + bar_width * (len(algorithms) - 1) / 2, filenames, rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison_costs.png')
    plt.show()


# =============================================================================
#                               Runner
# =============================================================================
class ExactAlgorithmMatrixRunner:

    def run_on_folder(self, folder: str):
        algorithms: Dict[str, Callable[[list], tuple]] = {
            'Branch & Bound': lambda matrix: ExactTSPSolver().branch_and_bound_solver(matrix),
            'Brute Force': lambda matrix: ExactTSPSolver().brute_force_solver(matrix),
            'Held-Karp': lambda matrix: ExactTSPSolver().held_karp_solver(matrix),
        }

        results_by_algorithm = {name: {'times': [], 'costs': []} for name in algorithms}
        city_nums = []
        filenames = []

        for filename in sorted(os.listdir(folder)):
            if not filename.endswith('.txt'):
                continue

            try:
                matrix = read_distance_matrix_from_txt(os.path.join(folder, filename))
            except Exception as e:
                print(f"Failed to read {filename}: {e}")
                continue

            city_num = len(matrix)

            print(f"\nRunning {filename} ({city_num} cities)...")
            city_nums.append(city_num)
            filenames.append(filename)

            for algo_name, algo_func in algorithms.items():
                try:
                    start = time.time()
                    cost, _ = algo_func(matrix)
                    duration = time.time() - start

                    results_by_algorithm[algo_name]['times'].append(duration)
                    results_by_algorithm[algo_name]['costs'].append(cost)

                    print(f"  [{algo_name}] Cost: {cost}, Time: {duration:.4f}s")
                except Exception as e:
                    print(f"  [{algo_name}] failed: {e}")

        # Sort all results by city count
        sort_index = np.argsort(city_nums)#生成一个排序索引数组，告诉你 city_nums 应该按什么顺序排列。
        city_nums = np.array(city_nums)[sort_index]#将城市数列表按正确顺序重新排列
        filenames = np.array(filenames)[sort_index]#同步对文件名排序，保持和 city_nums 的顺序一致。

        for algo_name in algorithms:
            for key in ['times', 'costs']:
                values = results_by_algorithm[algo_name][key]
                if len(values) != len(sort_index):
                    print(f"Skipping sorting for {algo_name} - {key} due to length mismatch.")
                    continue
                results_by_algorithm[algo_name][key] = np.array(values)[sort_index]

        # PLOT
        plot_running_time_vs_number_of_cities(city_nums, results_by_algorithm, algorithms)
        plot_cost_comparison(filenames, results_by_algorithm, algorithms)

if __name__ == "__main__":
    runner = ExactAlgorithmMatrixRunner()
    runner.run_on_folder("exact_data")
