import networkx as nx
import numpy as np
# Easy to work with graphs
from networkx.algorithms.tree import minimum_spanning_tree
from networkx.algorithms.matching import min_weight_matching
from networkx.algorithms.matching import max_weight_matching
import math


class ApproximateTSPSolver:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.n = len(distance_matrix) - 1
        self.best_cost = math.inf
        self.best_path = []

    def christofides_solver(self):
        import networkx as nx
        from networkx.algorithms.tree import minimum_spanning_tree
        from networkx.algorithms.matching import min_weight_matching

        print("Start Christofides Solver")
        n = self.n
        distance_matrix = self.distance_matrix

        G = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=distance_matrix[i][j])

        # Step 1: Minimum Spanning Tree
        mst = minimum_spanning_tree(G)

        # Step 2: Odd-degree vertices
        odd_nodes = [v for v in mst.nodes if mst.degree[v] % 2 == 1]

        # Step 3: Minimum weight matching between odd-degree vertices
        odd_subgraph = G.subgraph(odd_nodes)
        matching = min_weight_matching(odd_subgraph)  # ✅ no keyword args
        print(f"Found matching: {len(matching)} edges for {len(odd_nodes)} odd nodes")

        if len(matching) < len(odd_nodes) // 2:
            print("⚠️ Warning: Matching is not perfect, skipping Christofides.")
            return float('inf'), []

        # Step 4: Add matching to MST → multigraph
        multi = nx.MultiGraph(mst)
        for u, v in matching:
            multi.add_edge(u, v, weight=distance_matrix[u][v])

        # Step 5: Eulerian circuit
        if not nx.is_eulerian(multi):
            print("❌ Not Eulerian even after matching.")
            return float('inf'), []

        euler_circuit = list(nx.eulerian_circuit(multi))

        # Step 6: Shortcut to Hamiltonian cycle
        visited = set()
        path = []
        for u, _ in euler_circuit:
            if u not in visited:
                visited.add(u)
                path.append(u)
        path.append(path[0])

        cost = 0
        for i in range(len(path) - 1):
            cost += distance_matrix[path[i]][path[i + 1]]

        print("Finish Christofides Solver")
        return cost, path

    def nearest_neighbors_solver(self):

        n = self.n
        distance_matrix = self.distance_matrix
        visited = [False] * n
        path = [0]
        visited[0] = True
        total_cost = 0

        current_city = 0
        for _ in range(n - 1):
            # Find the nearest unvisited city
            nearest_city = None
            nearest_distance = float("inf")
            for next_city in range(n):
                if not visited[next_city] and distance_matrix[current_city][next_city] < nearest_distance:
                    nearest_city = next_city
                    nearest_distance = distance_matrix[current_city][next_city]

            # Visit the nearest city
            path.append(nearest_city)
            visited[nearest_city] = True
            total_cost += nearest_distance
            current_city = nearest_city

        total_cost += distance_matrix[current_city][0]
        path.append(0)

        self.route = path
        self.cost = total_cost

        return total_cost, path
