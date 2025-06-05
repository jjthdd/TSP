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
        # 生成从城市1到城市n-1的所有排列组合
        # 不包括城市0（因为它固定是起点和终点）
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
        #     如果你没有 dp[{0, i}, i]，就根本没法构造 dp[{0, i, j}, j]

        # Iterate over subsets of increasing size
        for subset_size in range(2, n):
            for subset in itertools.combinations(range(1, n), subset_size):#subset_size表示路径中除了城市0之外，访问了几个城市
                subset = frozenset(subset) | {0}  # Include the start node
                for current_node in subset - {0}:
                    # Calculate the minimum cost to reach the current node
                    dp[(subset, current_node)] = min(
                        dp[(subset - {current_node}, prev_node)] + distance_matrix[prev_node][current_node]
                        for prev_node in subset if prev_node != 0 and prev_node != current_node
                    )
        #             每一项代表一条候选路径：
        #
        # 先走到 prev_node（状态：dp[subset - {current_node}, prev_node]）
        #
        # 再从 prev_node 走到 current_node（花费 distance[prev_node][current_node]）
        #
        # 程序试所有 prev_node，选择花费最小的路径 说白了 先走回之前那个城市计算出的是最小cost 再加上到现在在的那个城市的cost的最小值加起来也是最小值
        # - pseudocode
        # opt := mink≠1 [g({2, 3, ..., n}, k) + d(k, 1)]
        # Find the minimum cost to complete the tour and return to the start
        full_set = frozenset(range(n))
        min_cost = min(
            dp[(full_set, current_node)] + distance_matrix[current_node][0]
            for current_node in range(1, n)
        )
        # 回到城市0形成完整路径
        # 我们最终访问了所有城市 full_set
        #
        # 尝试以每个城市作为最后一站 current_node，再加上 → 0 的回程
        #
        # 所有组合中花费最小的，就是完整 TSP 的最优解

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
        # 从最后一步开始，逐步回推出“上一步是谁”
        #
        # 把路径顺序还原出来
        #
        # 最终得到从0出发，访问所有城市，回到0 的完整路径列表

        return min_cost, path
# Held-Karp 每一层都会用“较小子集的最优解”去构造“更大子集的最优解”
#
# 每次都是在尝试 所有可能的转移路径 并 取最小值