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


def plot_cost_comparison(filenames, results_by_algorithm, algorithms, optimal_solutions):
    if optimal_solutions:
        opt_values = np.array([optimal_solutions.get(name, 0) for name in filenames])
        bar_width = 0.2
        x = np.arange(len(filenames))
        plt.figure(figsize=(12, 6))
        plt.bar(x - bar_width, opt_values, width=bar_width, label="Optimal")
        for algo_name, offset in zip(algorithms, range(1, len(algorithms)+1)):
            plt.bar(x - bar_width + offset * bar_width, results_by_algorithm[algo_name]['costs'],
                    width=bar_width, label=algo_name)
        plt.xlabel("Instance File")
        plt.ylabel("Tour Cost")
        plt.title("Tour Cost: Computed vs Optimal")
        plt.xticks(x, filenames, rotation=30)
        plt.legend()
        plt.tight_layout()
        plt.savefig('comparison_costs.png')
        plt.show()


def plot_gap_comparison(filenames, results_by_algorithm, algorithms):
    plt.figure(figsize=(12, 6))
    for algo_name in algorithms:
        gaps = results_by_algorithm[algo_name]['gaps']
        plt.plot(filenames, gaps, marker='x', label=algo_name)
    plt.xlabel("Instance File")
    plt.ylabel("Gap (Computed - Optimal)")
    plt.title("Error Gap per Algorithm")
    plt.xticks(rotation=30)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison_gaps.png')
    plt.show()
# =============================================================================
#                               Runner
# =============================================================================
class ExactAlgorithmMatrixRunner:

    def run_on_folder(self, folder: str, optimal_solutions: Dict[str, float] = None):
        algorithms: Dict[str, Callable[[list], tuple]] = {
            'Branch & Bound': lambda matrix: ExactTSPSolver().branch_and_bound_solver(matrix),
            'Brute Force': lambda matrix: ExactTSPSolver().brute_force_solver(matrix),
            'Held-Karp': lambda matrix: ExactTSPSolver().held_karp_solver(matrix),
        }

        results_by_algorithm = {name: {'times': [], 'costs': [], 'gaps': []} for name in algorithms}
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
            opt = optimal_solutions.get(filename, None)

            print(f"\nRunning {filename} ({city_num} cities)...")
            city_nums.append(city_num)
            filenames.append(filename)

            for algo_name, algo_func in algorithms.items():
                start = time.time()
                cost, _ = algo_func(matrix)#只要第一个值cost 第二个值path用不到
                duration = time.time() - start

                gap = cost - opt if opt is not None else None
                results_by_algorithm[algo_name]['times'].append(duration)
                results_by_algorithm[algo_name]['costs'].append(cost)
                results_by_algorithm[algo_name]['gaps'].append(gap)

                print(f"  [{algo_name}] Cost: {cost}, Time: {duration:.4f}s, Gap: {gap if gap is not None else 'N/A'}")

        # Sort all results by city count
        sort_index = np.argsort(city_nums)
        city_nums = np.array(city_nums)[sort_index]
        filenames = np.array(filenames)[sort_index]

        for algo_name in algorithms:
            for key in ['times', 'costs', 'gaps']:
                results_by_algorithm[algo_name][key] = np.array(results_by_algorithm[algo_name][key])[sort_index]

        # PLOT
        plot_running_time_vs_number_of_cities(city_nums, results_by_algorithm, algorithms)
        plot_cost_comparison(filenames, results_by_algorithm, algorithms, optimal_solutions)
        plot_gap_comparison(filenames, results_by_algorithm, algorithms)



if __name__ == "__main__":
    optimal_solutions = {
        "city5.txt": 19,
       }
    runner = ExactAlgorithmMatrixRunner()
    runner.run_on_folder("exact_data", optimal_solutions)
