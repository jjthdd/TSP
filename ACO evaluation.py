import os
import time
import numpy as np
import matplotlib.pyplot as plt
import csv

from AntColonyOptimizationAlgorithm import AntColony, read_distance_matrix, compute_error_rate


# 读取 solutions.txt 文件
def load_optimal_solutions(path='example/solutions.txt'):
    optimal = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' in line:
                name, val = line.strip().split(':')
                optimal[name.strip()] = float(val.strip())
    return optimal



def multi_example_evaluation(example_folder, solutions_file, param_name, param_values, base_params, num_runs=10):
    solutions = load_optimal_solutions()
    results = {}  # {example_name: {param_val: {'error_rates':[], ...}}}

    for example_file, optimal_dist in solutions.items():
        example_path = os.path.join(example_folder, example_file + '.txt')
        print(example_path)
        distance_matrix = read_distance_matrix(example_path)
        example_name = example_file

        results[example_name] = {}
        print(f"\n=== Example: {example_name} (Optimal={optimal_dist}) ===")

        for val in param_values:
            first_iters = []
            error_rates = []
            run_times = []

            for run in range(num_runs):
                print(f"\n--- {example_name} {param_name}={val}, Run {run + 1}/{num_runs} ---")
                params = base_params.copy()
                params[param_name] = val

                aco = AntColony(
                    distance_matrix,
                    num_ants=params['num_ants'],
                    num_iterations=params['num_iterations'],
                    alpha=params['alpha'],
                    beta=params['beta'],
                    evaporation=params['evaporation'],
                    Q=params['Q']
                )

                start_time = time.time()
                best_path, best_dist, first_iter = aco.run()
                elapsed_time = time.time() - start_time

                error_rate = compute_error_rate(best_dist, optimal_dist)

                first_iters.append(first_iter)
                error_rates.append(error_rate)
                run_times.append(elapsed_time)

            results[example_name][val] = {
                'first_iters': first_iters,
                'error_rates': error_rates,
                'run_times': run_times
            }

    # ===== Visualization=====
    for metric in ['error_rates', 'first_iters', 'run_times']:
        plt.figure(figsize=(12, 6))
        for example_name in results.keys():
            data = [results[example_name][val][metric] for val in param_values]
            pos_offset = list(results.keys()).index(example_name) * 0.2
            plt.boxplot(data,
                        positions=[i + 1 + pos_offset for i in range(len(param_values))],
                        widths=0.15,
                        patch_artist=True,
                        boxprops=dict(facecolor="lightblue"),
                        labels=[str(val) for val in param_values] if example_name == list(results.keys())[0] else [
                                                                                                                      ""] * len(
                            param_values),
                        )
        plt.title(f"{metric.replace('_', ' ').title()} vs {param_name} (across examples)")
        plt.xlabel(param_name)
        plt.ylabel(metric.replace('_', ' ').title())
        plt.grid(True)
        plt.legend(results.keys(), loc='upper right')
        plt.tight_layout()
        plt.show()

    # ===== 保存 CSV =====
    with open(f"aco_multi_eval_{param_name}.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['example', 'param_value', 'run', 'first_iter', 'error_rate', 'run_time'])
        for example_name in results.keys():
            for val in param_values:
                for i in range(num_runs):
                    writer.writerow([
                        example_name,
                        val,
                        i + 1,
                        results[example_name][val]['first_iters'][i],
                        results[example_name][val]['error_rates'][i],
                        results[example_name][val]['run_times'][i]
                    ])
    print("All results saved to CSV.")


if __name__ == '__main__':
    base_params = {
        'num_ants': 10,
        'num_iterations': 7,
        'alpha': 1.0,
        'beta': 5.0,
        'evaporation': 0.5,
        'Q': 100
    }

    multi_example_evaluation(
        example_folder='example',
        solutions_file='example/solutions.txt',
        param_name='evaporation',
        param_values=[0.1, 0.3, 0.5, 0.7, 0.9],
        base_params=base_params,
        num_runs=10
    )
