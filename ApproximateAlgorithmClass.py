import networkx as nx
# Easy to work with graphs
from networkx.algorithms.tree import minimum_spanning_tree
from networkx.algorithms.matching import max_weight_matching
import math


class ApproximateTSPSolver:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.n = len(distance_matrix) - 1
        self.best_cost = math.inf
        self.best_path = []

    def christofides_solver(self):

        graph = nx.Graph()
        n = self.n
        distance_matrix = self.distance_matrix
        # Add edges to the graph with weights
        for i in range(n):
            for j in range(i + 1, n):
                print(i, j)
                graph.add_edge(i, j, weight=distance_matrix[i][j])

        # Step 1: Find the Minimum Spanning Tree (MST)
        mst = minimum_spanning_tree(graph)

        # Step 2: Find vertices with odd degree in the MST
        odd_degree_nodes = [node for node in mst.nodes if mst.degree[node] % 2 == 1]

        # Step 3: Find a Minimum Weight Perfect Matching on the odd-degree vertices
        odd_graph = nx.Graph()
        for i in range(len(odd_degree_nodes)):
            for j in range(i + 1, len(odd_degree_nodes)):
                u, v = odd_degree_nodes[i], odd_degree_nodes[j]
                weight = -distance_matrix[u][v]  # Negative for max_weight_matching
                odd_graph.add_edge(u, v, weight=weight)

        matching = max_weight_matching(odd_graph, maxcardinality=True)

        # Step 4: Combine the MST and the matching edges
        mst_graph = nx.MultiGraph(mst)
        mst_graph.add_edges_from(matching)

        # Step 5: Form an Eulerian circuit
        eulerian_circuit = list(nx.eulerian_circuit(mst_graph))

        # Step 6: Shortcut to create the Hamiltonian circuit
        visited = set()
        hamiltonian_circuit = []
        for u, v in eulerian_circuit:
            if u not in visited:
                visited.add(u)
                hamiltonian_circuit.append(u)
        hamiltonian_circuit.append(hamiltonian_circuit[0])  # Return to the start

        total_cost = 0
        for i in range(len(hamiltonian_circuit) - 1):
            total_cost += distance_matrix[hamiltonian_circuit[i]][hamiltonian_circuit[i + 1]]

        self.best_path = hamiltonian_circuit
        self.best_cost = total_cost

        return total_cost, hamiltonian_circuit

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
#
# if __name__ == '__main__':
#     # Example distance matrix
#     distance_matrix = [
#         [0, 2, 9, 10],
#         [1, 0, 6, 4],
#         [15, 7, 0, 8],
#         [6, 3, 12, 0]
#     ]
#     solver = ApproximateTSPSolver(4, None)
#     cost, path = solver.christofides_tsp(distance_matrix)
#
#     print("Cost:", cost)
#     print("Path:", path)
#     cost, path = solver.nearest_neighbors(distance_matrix)
#     print("Cost:", cost)
#     print("Path:", path)
