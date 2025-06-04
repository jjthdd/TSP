import numpy as np
import random
import matplotlib.pyplot as plt


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

    def run(self):
        best_path = None
        best_distance = float('inf')
        # The number of iteration is 50 here
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

            self.evaporate_pheromones()
            self.deposit_pheromones(all_paths)
            self.history.append((best_path.copy(), best_distance))
            print(f"Iteration {iteration + 1}/{self.num_iterations}, Best Distance: {best_distance:.2f}")

        return best_path, best_distance

    # All ants are tasked with reaching the destination node

    # in a finite number of steps.(Not limited)
    def construct_solution(self):
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
        # Construct the whole path
        while len(path) < self.num_cities:
            next_city = self.select_next_city(current_city, visited)
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
        probabilities = []
        for city in range(self.num_cities):
            if city in visited: # Skipping vistied cities
                probabilities.append(0)
            else:
                pheromone = self.pheromone[current][city] ** self.alpha #\tau_{ij}^{\alpha}
                print("pheromone[current][city]", pheromone)
                heuristic = (1 / self.distances[current][city]) ** self.beta #\eta_{ij}^{\beta}
                print("heuristic[current][city]", heuristic)
                probabilities.append(pheromone * heuristic)

        probabilities = np.array(probabilities)
        print(probabilities)

        # When the probability of selection of all candidate cities is 0 (possibly with a distance of inf),
        # an unvisited city is randomly selected
        if probabilities.sum() == 0:
            return random.choice([i for i in range(self.num_cities) if i not in visited])
        probabilities /= probabilities.sum() # Normalization
        print("The probability of selecting next city is {}", probabilities)
        return np.random.choice(range(self.num_cities), p=probabilities)

    def path_distance(self, path):
        return sum(self.distances[path[i]][path[i + 1]] for i in range(len(path) - 1))

    # 信息素每轮都会蒸发一部分（防止过度集中）。
    # 距离相同情况下走费洛蒙高的路径 best path

    def evaporate_pheromones(self):
        self.pheromone *= (1 - self.evaporation)
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

# 使用方法
distance_matrix = read_distance_matrix("example/d1655.txt")

aco = AntColony(distance_matrix, num_ants=20, num_iterations=50)
best_path, best_dist = aco.run()

print("Best path:", best_path)
print("Best distance:", best_dist)

aco.plot_pheromone_heatmap()
aco.plot_convergence()
