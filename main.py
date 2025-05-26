import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import time
import json
import seaborn as sns
import pandas as pd
from ApproximateAlgorithm.ApproximateAlgorithmClass import ApproximateTSPSolver
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# =============================================================================
#                               Evaluation Metrics
# =============================================================================

def evaluate_ml_metrics(df):
    results = []

    for algo in df["Algorithm"].unique():
        df_algo = df[df["Algorithm"] == algo]
        y_true = df_algo["Optimal"].values
        y_pred = df_algo["Cost"].values

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = (np.abs(y_true - y_pred) / y_true).mean() * 100
        r2 = r2_score(y_true, y_pred)

        results.append({
            "Algorithm": algo,
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "MAPE (%)": round(mape, 2),
            "R2 Score": round(r2, 4)
        })

    return pd.DataFrame(results)

# =============================================================================
#                               Plot Methods
# =============================================================================

def plot_actual_vs_predicted_bar(df, save_path="plots/actual_vs_predicted_bar.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 准备数据（转换为长格式）
    melt_df = df[["Filename", "Algorithm", "Cost", "Optimal"]].copy()
    melt_opt = melt_df[["Filename", "Optimal"]].drop_duplicates()
    melt_opt["Algorithm"] = "Optimal"
    melt_opt.rename(columns={"Optimal": "Cost"}, inplace=True)

    combined = pd.concat([
        melt_df[["Filename", "Algorithm", "Cost"]],
        melt_opt
    ])

    combined = combined.sort_values("Filename")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=combined, x="Filename", y="Cost", hue="Algorithm")
    plt.xticks(rotation=45, ha="right")
    plt.yscale("log")
    plt.title("Comparison of Optimal vs Algorithm Costs")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved as {save_path}")

def plot_runtime_vs_size(size_list, time_list, method_name):
    plt.figure()
    if len(size_list) != len(time_list):
        print(
            f"Size mismatch for method {method_name}: size_list has {len(size_list)}, time_list has {len(time_list)}")
    else:
        plt.plot(size_list, time_list, marker='o')
    plt.title(f"{method_name} Solver Runtime vs Matrix Size")
    plt.xlabel("Matrix Size (n x n)")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =============================================================================
#                               Load Result
# =============================================================================
def load_optimal_solutions(path='example/solutions.txt'):
    optimal = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' in line:
                name, val = line.strip().split(':')
                optimal[name.strip()] = float(val.strip())
    return optimal


def evaluate_algorithms(result_json_path='result_summary.json', solution_path='example/solutions.txt'):
    with open(result_json_path, 'r') as f:
        results = json.load(f)
    optimal = load_optimal_solutions(solution_path)

    records = []

    for fname, algos in results.items():
        base_name = os.path.splitext(fname)[0]  # ✅ 去掉 .txt
        if base_name not in optimal:
            print(f"Skip {fname}, no optimal solution in file solutions.txt")
            print("http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html")
            continue

        opt_cost = optimal[base_name]
        for algo_name, vals in algos.items():
            cost = vals['cost']
            time_sec = vals['time']
            abs_err = cost - opt_cost
            rel_err = (abs_err / opt_cost) * 100

            records.append({
                "Filename": fname,
                "Algorithm": algo_name,
                "Cost": cost,
                "Optimal": opt_cost,
                "Absolute Error": abs_err,
                "Relative Error (%)": rel_err,
                "Time (s)": time_sec
            })
    df = pd.DataFrame(records)
    df.to_csv("evaluation_results.csv", index=False)
    print("✅ Evaluation saved to evaluation_results.csv")

    return df



# =============================================================================
#                               Read and Convert TSPLIB
# =============================================================================
class MatrixGenerator:
    def __init__(self):
        self.n = 0

    def read_file(self):
        all_matrices = {}
        txt_files = glob.glob(os.path.join('example', "*.txt"))

        for filename in txt_files:
            if os.path.basename(filename) == "solutions.txt":
                continue  #  跳过答案文件

            with open(filename, 'r') as f:
                matrix = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue  # 跳过空行
                    row = list(map(float, line.split()))
                    matrix.append(row)
                all_matrices[filename] = matrix

        return all_matrices

    def tsp_to_distance_matrix(self, coords: dict, mode='EUC_2D'):
        print(f"Converting tsp coordinates to distance matrix using mode: {mode}")
        n = len(coords)
        matrix = np.zeros((n, n))
        coord_list = [coords[i + 1] for i in range(n)]  # TSPLIB是1-indexed

        for i in range(n):
            for j in range(n):
                if i != j:
                    if mode == 'EUC_2D':
                        xi, yi = coord_list[i]
                        xj, yj = coord_list[j]
                        distance = np.linalg.norm(np.array([xi, yi]) - np.array([xj, yj]))
                    elif mode == 'GEO':
                        distance = self.geo_distance(coord_list[i], coord_list[j])
                    elif mode == 'ATT':
                        distance = self.att_distance(coord_list[i], coord_list[j])
                    elif mode == 'CEIL_2D':
                        xi, yi = coord_list[i]
                        xj, yj = coord_list[j]
                        distance = np.ceil(np.linalg.norm(np.array([xi, yi]) - np.array([xj, yj])))
                    else:
                        raise ValueError(f"Unsupported distance mode: {mode}")
                    matrix[i][j] = distance
        return matrix.tolist()

    def geo_distance(self, coord1, coord2):
        import math
        def to_radians(deg_min):
            deg = int(deg_min)
            min_ = deg_min - deg
            return math.pi * (deg + 5.0 * min_ / 3.0) / 180.0

        RRR = 6378.388
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        lat1, lon1 = to_radians(lat1), to_radians(lon1)
        lat2, lon2 = to_radians(lat2), to_radians(lon2)

        q1 = math.cos(lon1 - lon2)
        q2 = math.cos(lat1 - lat2)
        q3 = math.cos(lat1 + lat2)

        dij = int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
        return dij

    def att_distance(self, coord1, coord2):
        xi, yi = coord1
        xj, yj = coord2
        dx = xi - xj
        dy = yi - yj
        rij = np.sqrt((dx * dx + dy * dy) / 10.0)
        tij = int(rij + 0.5)  # round to nearest integer
        return tij

    def write_matrix_to_txt(self, matrix, out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            for row in matrix:
                row_str = ' '.join(map(lambda x: f"{x:.4f}", row))
                f.write(row_str + '\n')
        print(f"Saved matrix to: {out_path}")

    def read_tsp_files(self, folder='tsplib'):
        all_matrices = {}
        output_data = {}

        tsp_files = glob.glob(os.path.join(folder, "*.tsp"))
        print("Start reading tsp files...")

        for filename in tsp_files:
            print(f"Reading file: {filename}")
            meta_info, matrix_or_coords = self.parse_tsp_file(filename)

            etype = meta_info.get("EDGE_WEIGHT_TYPE")
            if etype == "EUC_2D":
                matrix = self.tsp_to_distance_matrix(matrix_or_coords, mode='EUC_2D')
            elif etype == "GEO":
                matrix = self.tsp_to_distance_matrix(matrix_or_coords, mode='GEO')
            elif etype == "ATT":
                matrix = self.tsp_to_distance_matrix(matrix_or_coords, mode='ATT')
            elif etype == "EXPLICIT":
                  # and meta_info.get("EDGE_WEIGHT_FORMAT") == "FULL_MATRIX")  and UPPER_ROW:
                matrix = matrix_or_coords
            elif etype == "CEIL_2D":
                matrix = self.tsp_to_distance_matrix(matrix_or_coords, mode='CEIL_2D')
            else:
                raise ValueError(f"Unsupported format in file {filename} with EDGE_WEIGHT_TYPE {etype}")

            # 写入 .txt 文件
            basename = os.path.splitext(os.path.basename(filename))[0]
            out_path = os.path.join('example', f"{basename}.txt")
            self.write_matrix_to_txt(matrix, out_path)

            # 存入 dict
            all_matrices[filename] = matrix
            output_data[basename] = {
                "matrix": matrix,
                "meta": meta_info
            }

        print("Finish reading tsp files")

        with open('output.json', 'w') as outfile:
            json.dump(output_data, outfile, indent=2)

        return all_matrices

    def parse_tsp_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        meta_info = {}
        coords = {}
        matrix = []
        n = 0

        for line in lines:
            if line.startswith("EDGE_WEIGHT_TYPE"):
                meta_info["EDGE_WEIGHT_TYPE"] = line.split(":")[-1].strip()
            elif line.startswith("EDGE_WEIGHT_FORMAT"):
                meta_info["EDGE_WEIGHT_FORMAT"] = line.split(":")[-1].strip()
            elif line.startswith("DIMENSION"):
                n = int(line.split(":")[-1].strip())
            elif line == "NODE_COORD_SECTION" or line == "EDGE_WEIGHT_SECTION":
                break

        if meta_info.get("EDGE_WEIGHT_TYPE") in ["EUC_2D", "GEO", "ATT", "CEIL_2D"]:
            in_coord_section = False
            for line in lines:
                if line == "NODE_COORD_SECTION":
                    in_coord_section = True
                    continue
                if line == "EOF":
                    break
                if in_coord_section:
                    parts = line.split()
                    if len(parts) >= 3:
                        idx = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        coords[idx] = (x, y)
            return meta_info, coords


        elif meta_info.get("EDGE_WEIGHT_TYPE") == "EXPLICIT":
            ew_format = meta_info.get("EDGE_WEIGHT_FORMAT")
            in_matrix_section = False
            flat_numbers = []
            for line in lines:
                if line == "EDGE_WEIGHT_SECTION":
                    in_matrix_section = True
                    continue
                if line in ["EOF", "DISPLAY_DATA_SECTION", "NODE_COORD_SECTION", "DISPLAY_SECTION"]:
                    break
                if in_matrix_section:
                    parts = line.split()
                    try:
                        flat_numbers.extend(map(float, parts))
                    except ValueError:
                        print(f"⚠️: Skipping non-numeric line in {file_path}: {line}")
                        break
            if ew_format == "FULL_MATRIX":
                matrix = [flat_numbers[i * n:(i + 1) * n] for i in range(n)]
                return meta_info, matrix
            elif ew_format == "UPPER_ROW" or ew_format == "UPPER_DIAG_ROW":
                matrix = [[0.0 for _ in range(n)] for _ in range(n)]
                index = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        matrix[i][j] = flat_numbers[index]
                        matrix[j][i] = flat_numbers[index]  # 对称矩阵
                        index += 1
                return meta_info, matrix
            elif ew_format == "LOWER_DIAG_ROW":
                matrix = [[0.0 for _ in range(n)] for _ in range(n)]
                index = 0
                for i in range(n):
                    for j in range(i + 1):  # 包括对角线
                        matrix[i][j] = flat_numbers[index]
                        matrix[j][i] = flat_numbers[index]  # 对称赋值
                        index += 1
                return meta_info, matrix
            else:
                    raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT '{ew_format}' in file {file_path}")

    def print_result(self, name, cost, path, elapsed):
        print(f"The name of algorithm: {name}")
        print(f"The best cost: {cost}")
        print(f"The best path: {path}")
        print(f"Cost Time: {elapsed:.4f} seconds\n")




if __name__ == '__main__':
    # =================================Run Once =============================
    generator = MatrixGenerator()
    print("Reading File List")
    generator.read_tsp_files()
    matrices_map = generator.read_file()  # Read example/*.txt

    filename_list = matrices_map.keys()

    print(filename_list)

    # won't add exact algorithm
    result_data = {}
    methods = {
        "Nearest Neighbor": lambda m: ApproximateTSPSolver(m).nearest_neighbors_solver(),
        "Christofides Solver": lambda m: ApproximateTSPSolver(m).christofides_solver()
    }

    os.makedirs("plots", exist_ok=True)

    # Save approximate result
    size_lists = {method: [] for method in methods}
    time_lists = {method: [] for method in methods}

    # throw exceptions
    def is_square_matrix(matrix):
        n = len(matrix)
        return all(len(row) == n for row in matrix)


    for filename in filename_list:
        base_name = os.path.basename(filename)
        print(f"\n Solving {base_name}")
        input_matrix = matrices_map[filename]

        if not is_square_matrix(input_matrix):
            print(f"Skipping {filename}: matrix is not square.")
            continue
        size = len(input_matrix)

        result_data[base_name] = {}

        for method_name, solve_fn in methods.items():
            print(f" Running {method_name}...")
            start_time = time.time()
            cost, path = solve_fn(input_matrix)
            elapsed = time.time() - start_time

            generator.print_result(method_name, cost, path, elapsed)

            result_data[base_name][method_name] = {
                "cost": cost,
                "path": path,
                "time": elapsed,
                "size": size
            }

            size_lists[method_name].append(size)
            time_lists[method_name].append(elapsed)

    #  Save result as json
    with open("result_summary.json", "w") as f:
        json.dump(result_data, f, indent=2)
    # save result as plot
    for method_name in methods:
        sizes = size_lists[method_name]
        times = time_lists[method_name]

        # Sort size
        sorted_pairs = sorted(zip(sizes, times), key=lambda x: x[0])
        sorted_sizes, sorted_times = zip(*sorted_pairs)

        plt.figure()
        plt.plot(sorted_sizes, sorted_times, marker='o')
        plt.title(f"{method_name} Solver Runtime vs Matrix Size (Sorted)")
        plt.xlabel("Matrix Size (n x n)")
        plt.ylabel("Runtime (seconds)")
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join("plots", f"{method_name}_runtime_sorted.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Runtime plot saved as {plot_path}")

    # ============================================
    # Evaluation - result stored in result_summary.json
    df = evaluate_algorithms()
    plot_actual_vs_predicted_bar(df)
    result_df = evaluate_ml_metrics(df)
    print(result_df)


