import math
import itertools
from typing import List, Tuple, Any


class ExactTSPSolver:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.n = len(distance_matrix)
        self.best_cost = math.inf
        self.best_path = []
        """
            branch and bound —— instead of permuting all the possibilities, divide the problem into
            subquestion and find its lower bound.  
        """
    # Exhaustive, iterate over all pathss
    def brute_force_solver(self) -> Tuple[float, list[Any]]:
        permutation = itertools.permutations(range(1, self.n))
        print(permutation)
        for perm in permutation:
            route = [0] + list(perm) + [0]
            # route = [0, 2, 3, 1, 0]
            print(route)
            cost = 0
            for i in range(len(route)- 1): # (0, 2, 3, 1, 0)
                print("cost:", self.distance_matrix[route[i]][route[i + 1]], "route: ", route[i], route[i+1])
                cost += self.distance_matrix[route[i]][route[i + 1]]
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_path = route
        return self.best_cost, self.best_path

    def branch_and_bound_solver(self) -> Tuple[float, list[Any]]:
        self.best_cost = float('inf')
        self.best_path = []

        def bound(current_path: List[int], visited: set) -> int:
            lower_bound = 0

            # already cost
            for i in range(len(current_path) - 1):
                lower_bound += self.distance_matrix[current_path[i]][current_path[i + 1]]
            print("lower_bound", lower_bound)
            for city in range(self.n):
                print(city)
                if city not in visited:
                    # For cities which are not visited, find the cheapest cost
                    min_cost = min(
                        self.distance_matrix[city][j] for j in range(self.n) if j != city
                    )
                    # Get the approximate result
                    lower_bound += min_cost
            return lower_bound
        # bfs: 0 -> 1, 0 -> 2, 0 -> 3
        # dfs: 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 0
        def dfs(current_path: List[int], visited: set, current_cost: int):
            current_city = current_path[-1]#-1是倒数第一个城市

            if len(current_path) == self.n:
                # If all cities are visited, then return city 0
                total_cost = current_cost + self.distance_matrix[current_city][current_path[0]]
                if total_cost < self.best_cost:
                    self.best_cost = total_cost
                    self.best_path = current_path[:]
                return

            for next_city in range(self.n):
                if next_city not in visited:
                    temp_cost = current_cost + self.distance_matrix[current_city][next_city]
                    temp_path = current_path + [next_city]
                    temp_visited = visited | {next_city} #|是添加的意思 因为visited是集合
                    # If it's estimated to minimum cost and current shortest route is larger, then cut it off
                    if bound(temp_path, temp_visited) < self.best_cost:
                        visited.add(next_city)
                        current_path.append(next_city)
                        dfs(current_path, visited, temp_cost)
                        visited.remove(next_city)
                        current_path.pop()

        dfs([0], {0}, 0)
        best_route = self.best_path + [0] if self.best_path else []
        return self.best_cost, best_route


# Example usage
if __name__ == "__main__":
    # Example distance matrix (symmetric matrix)
    distance_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]

    solver = ExactTSPSolver(distance_matrix)
    bb_best_cost, bb_best_path = solver.branch_and_bound_solver()
    bf_best_cost, bf_best_path = solver.brute_force_solver()
    #hk_best_cost, hk_best_path = solver.held_karp_solver()
    print("Branch and bound solution:")
    print("Best cost:", bb_best_cost)
    print("Best path:", bb_best_path)
    print("Brute force solution:")
    print("Best cost:", bf_best_cost)
    print("Best path:", bf_best_path)
    # print("Held-Karp solution:")
    # print("Best cost:", hk_best_cost)
    # print("Best path:", hk_best_path)
