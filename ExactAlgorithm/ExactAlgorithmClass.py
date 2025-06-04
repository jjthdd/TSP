import math
import itertools
from typing import List, Tuple, Any


class ExactTSPSolver:
    def __init__(self):
        self.best_cost = math.inf
        self.best_path = []
        """
            branch and bound —— instead of permuting all the possibilities, divide the problem into
            subquestion and find its lower bound.  
        """
    # Exhaustive, iterate over all paths
    def brute_force_solver(self, distance_matrix) -> Tuple[float, list[Any]]:
        n = len(distance_matrix)
        permutation = itertools.permutations(range(1, n))
        print("size:", n)
        for perm in permutation:
            route = [0] + list(perm) + [0]
            cost = 0
            # [0, 1, 2, 3, 0]
            for i in range(len(route) - 1):
                cost += distance_matrix[route[i]][route[i + 1]]
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_path = route

        return self.best_cost, self.best_path

    def branch_and_bound_solver(self, distance_matrix) -> Tuple[float, list[Any]]:
        self.best_cost = float('inf')
        self.best_path = []
        n = len(distance_matrix)
        def bound(current_path: List[int], visited: set) -> int:
            lower_bound = 0
            n = len(distance_matrix)
            # already cost
            for i in range(len(current_path) - 1):
                lower_bound += distance_matrix[current_path[i]][current_path[i + 1]]

            for city in range(n):

                if city not in visited:
                    # For cities which are not visited, find the cheapest cost
                    min_cost = min(
                        distance_matrix[city][j] for j in range(n) if j != city
                    )
                    # Get the approximate result
                    lower_bound += min_cost
            return lower_bound


        def dfs(current_path: List[int], visited: set, current_cost: int):
            current_city = current_path[-1]

            if len(current_path) == n:
                # If all cities are visited, then return city 0
                total_cost = current_cost + distance_matrix[current_city][current_path[0]]
                if total_cost < self.best_cost:
                    self.best_cost = total_cost
                    self.best_path = current_path[:]
                return

            for next_city in range(n):
                if next_city not in visited:
                    temp_cost = current_cost + distance_matrix[current_city][next_city]
                    temp_path = current_path + [next_city]
                    temp_visited = visited | {next_city}
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

    def held_karp_solver(self, distance_matrix):

        n = len(distance_matrix)

        dp = {}
        for i in range(1, n):  # Any other node

            dp[(frozenset([0, i]), i)] = distance_matrix[0][i]

        # Iterate over subsets of increasing size
        for subset_size in range(2, n):
            for subset in itertools.combinations(range(1, n), subset_size):
                subset = frozenset(subset) | {0}  # Include the start node
                for current_node in subset - {0}:
                    # Calculate the minimum cost to reach the current node
                    dp[(subset, current_node)] = min(
                        dp[(subset - {current_node}, prev_node)] + distance_matrix[prev_node][current_node]
                        for prev_node in subset if prev_node != 0 and prev_node != current_node
                    )
        # - pseudocode
        # opt := mink≠1 [g({2, 3, ..., n}, k) + d(k, 1)]
        # Find the minimum cost to complete the tour and return to the start
        full_set = frozenset(range(n))
        min_cost = min(
            dp[(full_set, current_node)] + distance_matrix[current_node][0]
            for current_node in range(1, n)
        )

        # Reconstruct the optimal path
        path = []
        subset = full_set
        last_node = 0

        for _ in range(n - 1, 0, -1):
            last_node = min(
                (dp[(subset, current_node)] + distance_matrix[current_node][last_node], current_node)
                for current_node in subset if current_node != 0 and current_node != last_node
            )[1]
            path.append(last_node)
            subset = subset - {last_node}

        path.append(0)
        path.reverse()

        return min_cost, path

# Example usage
# if __name__ == "__main__":
#     # Example distance matrix (symmetric matrix)
#     distance_matrix = [
#         [0, 10, 15, 20],
#         [10, 0, 35, 25],
#         [15, 35, 0, 30],
#         [20, 25, 30, 0]
#     ]
#
#     solver = ExactTSPSolver(distance_matrix)
#     bb_best_cost, bb_best_path = solver.branch_and_bound_solver()
#     bf_best_cost, bf_best_path = solver.brute_force_solver()
#     hk_best_cost, hk_best_path = solver.held_karp_solver()
#     print("Branch and bound solution:")
#     print("Best cost:", bb_best_cost)
#     print("Best path:", bb_best_path)
#     print("Brute force solution:")
#     print("Best cost:", bf_best_cost)
#     print("Best path:", bf_best_path)
#     print("Held-Karp solution:")
#     print("Best cost:", hk_best_cost)
#     print("Best path:", hk_best_path)
