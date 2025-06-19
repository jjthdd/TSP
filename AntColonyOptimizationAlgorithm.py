import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import time

class AntColony:
    def __init__(self, distance_matrix, num_ants, num_iterations, alpha=1.0, beta=5.0,
                 evaporation=0.5, Q=100, seed=42):
        np.random.seed(seed)
        random.seed(seed)

        self.distances = np.array(distance_matrix)
        self.num_cities = len(self.distances)
        self.pheromone = np.ones((self.num_cities, self.num_cities))
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q
        self.history = []
    # distance_matrix	城市之间的距离，二维表
    # num_ants	每次模拟多少只蚂蚁
    # num_iterations	一共模拟几轮（每轮所有蚂蚁都走一遍）
    # alpha / beta	控制蚂蚁对“信息素”和“距离”的敏感程度
    # evaporation	信息素蒸发率，防止信息素过多导致过拟合
    # Q	每只蚂蚁走完路径后，释放信息素的强度
    # pheromone	信息素矩阵，一开始所有路径信息素都是1
    # history	存每一轮的最好路径和距离，用来画图用

    def run(self):
        #主控函数，执行算法的核心逻辑
        #它做的是：
        # 重复若干次：
        #所有蚂蚁都找一条路径（construct_solution）
        # 选出最短路径
        # 信息素更新（先蒸发，再更新）
        best_path = None
        best_distance = float('inf')
        first_convergence_iteration = None  # 新增记录

        for iteration in range(self.num_iterations):
            all_paths = []
            print("Starting iteration {}".format(iteration))

            for _ in range(self.num_ants):
                path = self.construct_solution()
                distance = self.path_distance(path)
                all_paths.append((path, distance))

                if distance < best_distance:
                    best_distance = distance
                    best_path = path
                    if first_convergence_iteration is None:
                        first_convergence_iteration = iteration + 1  # 首次更新 best_distance 时记录

            self.evaporate_pheromones()
            self.deposit_pheromones(all_paths)
            self.history.append((best_path.copy(), best_distance))

            print(f"Iteration {iteration + 1}/{self.num_iterations}, Best Distance: {best_distance:.2f}")
            print(len(all_paths))

        return best_path, best_distance, first_convergence_iteration

    # All ants are tasked with reaching the destination node

    # in a finite number of steps.(Not limited)
    def construct_solution(self):
        # 构造一只蚂蚁的完整路径
        # 模拟一只蚂蚁怎么走：
        # 随机选择一个起点城市
        # 一步步选择下一个城市，直到走完所有城市
        # 最后返回起点（形成一个闭环）
        # 用到了下面那个函数来选“下一步去哪”：
        '''
        An ant starts with random city and select_next_city finally arrive the destination
        :return: The whole path List[Int]
        '''
        path = []
        visited = set()
        print("")
        current_city = random.randint(0, self.num_cities - 1) # Start with random city
        print("Starting path construction, the current_city is: ", current_city)
        path.append(current_city)
        visited.add(current_city)
        # 蚂蚁随机从某个城市出发，加入 path 和 visited
        # Construct the whole path
        while len(path) < self.num_cities:
            next_city = self.select_next_city(current_city, visited)
            # 不断选下一个没去过的城市，直到走完所有城市为止。每一步都依赖 select_next_city() 来决策去哪。
            print("Continue path construction, the next_city is: ", next_city)
            path.append(next_city)
            visited.add(next_city)
            current_city = next_city

        path.append(path[0])  # Return to starting city
        return path

    """ 

        p_{ij}^k =  
        \frac{(\tau_{ij})^{\alpha} (\eta_{ij})^{\beta}} 
        ——————————————————————————————————————————————————————————————————————
        {\sum_{z \in \text{allowed}_x} (\tau_{iz})^{\alpha} (\eta_{iz})^{\beta}}  ————  Sum of  Unvisited Cities 

        \tau_{ij}: pheromone concentration (pheromone[current][city])  - Lots of ants passed by the current city
        \eta_{ij}: distance concentration (distance[current][city])  - Ants prefer the shorter distance
        The denominator is a weighted sum of all selectable cities
    """
    def select_next_city(self, current, visited):
        # 决策怎么走下一步（城市还没走过，且更短的更容易选）
        # 通过“信息素 + 距离”的加权，算出每个候选城市的概率
        # 再用 np.random.choice() 随机选一个（但选概率高的可能性更大）
        probabilities = []
        epsilon = 1e-10
        for city in range(self.num_cities):
            if city in visited: # Skipping vistied cities
                probabilities.append(0)
                #筛掉已经去过的城市，这些城市在这一轮中不能被选中
            else:
                distance = self.distances[current][city]
                pheromone = self.pheromone[current][city] ** self.alpha #\tau_{ij}^{\alpha}
                # 取当前城市到该城市的 信息素值，指数化后作为权重（控制权重变化的强度由 alpha 决定）
                print("pheromone[current][city]", pheromone)
                heuristic = (1 / (distance + epsilon) ) ** self.beta #\eta_{ij}^{\beta}
                #取当前城市到该城市的 距离倒数，也做指数处理（距离越短，值越大）
                #beta 越大，表示蚂蚁越“贪心”，更看重短路径
                print("heuristic[current][city]", heuristic)
                probabilities.append(pheromone * heuristic)
                #计算出这个城市的综合吸引力 = 信息素 × 距离倒数

        probabilities = np.array(probabilities)
        print(probabilities)

        # When the probability of selection of all candidate cities is 0 (possibly with a distance of inf),
        # an unvisited city is randomly selected
        if probabilities.sum() == 0:
            return random.choice([i for i in range(self.num_cities) if i not in visited])
        #有时候所有城市的权重是0（比如某些距离是 ∞），就随机挑一个没走过的城市应急处理

        probabilities /= probabilities.sum() # Normalization
        #把所有值加起来变成1（标准概率分布），之后用np.random.choice() 按照比例选一个城市
        print("The probability of selecting next city is {}", probabilities)
        return np.random.choice(range(self.num_cities), p=probabilities)
       #信息素 (τ)	距离 (d)	权重 = τ^α * (1/d)^β

    def path_distance(self, path):
        return sum(self.distances[path[i]][path[i + 1]] for i in range(len(path) - 1))

    # 信息素每轮都会蒸发一部分（防止过度集中）。
    # 距离相同情况下走费洛蒙高的路径 best path

    def evaporate_pheromones(self):
        self.pheromone *= (1 - self.evaporation)
    # 这行代码的意思是：整张城市图上所有路径的信息素都乘以一个小于1的系数
    # self.evaporation 是蒸发率，比如 0.5 表示每轮留一半
    # 这相当于让旧信息逐渐消失，防止信息素“过度集中”在某条路径


    # 每只蚂蚁走完路径后，按照路径长度回馈信息素。
    # Each ant travels backward from the destination to the spawn_point and deposits pheromones on this path.
    # Q: strength of ant
    # L: distance
    def deposit_pheromones(self, all_paths):
        for path, L in all_paths:
            pheromone_amount = self.Q / L
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                self.pheromone[a][b] += pheromone_amount
                self.pheromone[b][a] += pheromone_amount  # Keep symmetric
    # 给路径上的每一条边加上信息素
    # 对路径上的每一段（a 到 b）加上刚才算出的信息素
    # 因为是对称的城市图，所以 a→b 和 b→a 都要加

    def plot_convergence(self):
        distances = [d for _, d in self.history]
        plt.figure(figsize=(6, 4))
        plt.plot(distances, marker='o', linestyle='-')
        plt.title("Best Distance Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Distance")
        plt.grid(True)
        plt.show()

    def plot_pheromone_heatmap(self):
        plt.figure(figsize=(6, 5))
        plt.imshow(self.pheromone, cmap='hot', interpolation='nearest')
        plt.colorbar(label="Pheromone Strength")
        plt.title("Pheromone Heatmap")
        plt.xlabel("City")
        plt.ylabel("City")
        plt.show()


# Example usage
def read_distance_matrix(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    matrix = []
    for line in lines:
        # 处理每一行的数字，忽略空行
        if line.strip():
            row = list(map(float, line.strip().split()))
            matrix.append(row)
    return np.array(matrix)
def compute_error_rate(best_dist, optimal_dist):
    return ((best_dist - optimal_dist) / optimal_dist) * 100


def experiment_param_benchmark(param_name, param_values, base_params, optimal_dist, num_runs=10):
    results = {}
    distance_matrix = read_distance_matrix("example/gr21.txt")

    for val in param_values:
        print(f"\n=== Testing {param_name}={val} ===")
        first_iters = []
        error_rates = []
        run_times = []

        for run in range(num_runs):
            print(f"\n--- Run {run + 1} ---")

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

            print(
                f"Run {run + 1}: First iter = {first_iter}, Best dist = {best_dist:.2f}, Error = {error_rate:.2f}%, Time = {elapsed_time:.2f}s")

            first_iters.append(first_iter)
            error_rates.append(error_rate)
            run_times.append(elapsed_time)

        results[val] = {
            'first_iters': first_iters,
            'error_rates': error_rates,
            'run_times': run_times
        }

    # save as CSV
    with open(f"aco_eval_{param_name}.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['param_value', 'run', 'first_iter', 'error_rate', 'run_time'])

        for val in param_values:
            for i in range(num_runs):
                writer.writerow([
                    val,
                    i + 1,
                    results[val]['first_iters'][i],
                    results[val]['error_rates'][i],
                    results[val]['run_times'][i]
                ])

    plt.figure(figsize=(8, 6))
    plt.boxplot([results[val]['error_rates'] for val in param_values],
                labels=[str(val) for val in param_values])
    plt.title(f"Effect of {param_name} on Solution Quality (Error Rate %)\n(num_runs={num_runs})")
    plt.xlabel(param_name)
    plt.ylabel("Error Rate (%)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.boxplot([results[val]['first_iters'] for val in param_values],
                labels=[str(val) for val in param_values])
    plt.title(f"Effect of {param_name} on First Convergence Iteration\n(num_runs={num_runs})")
    plt.xlabel(param_name)
    plt.ylabel("First Convergence Iteration")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.boxplot([results[val]['run_times'] for val in param_values],
                labels=[str(val) for val in param_values])
    plt.title(f"Effect of {param_name} on Run Time\n(num_runs={num_runs})")
    plt.xlabel(param_name)
    plt.ylabel("Run Time (s)")
    plt.grid(True)
    plt.show()
