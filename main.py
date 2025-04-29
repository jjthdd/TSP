import pandas as ps
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import time
from ExactAlgorithmClass import ExactTSPSolver
from ApproximateAlgorithmClass import ApproximateTSPSolver


class MatrixGenerator:
    def __init__(self):
        self.n = 0
        self.m = 0



    # already tried TSPLIB .tsp and transform matrix and generate json file (50000)
    def read_file(self):
        all_matrices = {}

        txt_files = glob.glob(os.path.join('example', "*.txt"))
        print(txt_files)

        for filename in txt_files:
            with open(filename, 'r') as f:
                matrix = []
                for line in f:
                    # 假设每一行由空格分隔
                    row = list(map(float, line.strip().split()))
                    matrix.append(row)
                all_matrices[filename] = matrix


        for fname, mat in all_matrices.items():
            print(f"file: {fname}")
            for row in mat:
                print(row)
            print("-" * 40)

        return all_matrices

    def print_result(self, name, cost, path, elapsed):
        print(f"The name of algorithm: {name}")
        print(f"The best cost: {cost}")
        print(f"The best path: {path}")
        print(f"Cost Time: {elapsed:.4f} seconds\n")


    def plot_runtime_vs_size(self, size_list, time_list, method_name):
        plt.figure()
        print(size_list, time_list)
        plt.plot(size_list, time_list, marker='o')
        plt.title(f"{method_name} Solver Runtime vs Matrix Size")
        plt.xlabel("Matrix Size (n x n)")
        plt.ylabel("Runtime (seconds)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':

    generator = MatrixGenerator()
    # mat_list = generator.generate_adjacency_matrices()
    matrices_map = generator.read_file()
    filename_list = matrices_map.keys()
    print("filename_list:", filename_list)
    size_list = []
    time_list = []

    for filename in filename_list:
        print("Start with " + filename)
        input_matrix = matrices_map[filename]
        size = len(input_matrix)
        exact_solver = ExactTSPSolver(input_matrix)
        approximate_solver = ApproximateTSPSolver(input_matrix)

        start_time = time.time()
        cost, path = approximate_solver.nearest_neighbors_solver()
        #cost, path = exact_solver.brute_force_solver()
        end_time = time.time()
        elapsed = end_time - start_time
        generator.print_result(filename, cost, path, elapsed)

        size_list.append(size)
        time_list.append(elapsed)

    generator.plot_runtime_vs_size(size_list, time_list, "Nearest Neighbor")
