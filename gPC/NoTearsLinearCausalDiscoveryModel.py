import logging
from tqdm import tqdm
import numpy as np
import networkx as nx
import pygpc
from castle.algorithms import Notears
from castle.datasets import DAG, IIDSimulation


# def simulate_equation(X, w, Z):
#     return X @ w + Z
    
# def simulate_dag(W, Z):
#     G =  nx.from_numpy_matrix(W, create_using=nx.DiGraph)
#     assert nx.is_directed_acyclic_graph(G), "W must represent a DAG"
#     X = np.zeros([Z.shape[1], Z.shape[0]])
#     ordered_vertices = list(nx.topological_sort(G))
#     for vertex in ordered_vertices:
#         parents = list(G.predecessors(vertex))
#         X[:, vertex] = simulate_equation(X[:, parents], W[parents, vertex], Z[vertex, :])
#     return X

# N_samples = 100
# N_nodes = 10
# N_edges = 20
# W_true = DAG.scale_free(n_nodes=N_nodes, n_edges=N_edges)
# Z = np.random.normal(loc=0, scale=1, size=(N_nodes, N_samples))

# X = simulate_dag(W_true, Z)


class NoTearsLinearCausalDiscoveryModel(pygpc.AbstractModel):

    def __init__(self, N_nodes, N_edges, N_learn_samples, N_res_samples, method, sem_type):
        self.N_nodes = N_nodes
        self.N_edges = N_edges
        self.N_learn_samples = N_learn_samples
        self.N_res_samples = N_res_samples
        self.method = method
        self.sem_type = sem_type
        self.model = Notears()
    
    def simulate(self, process_id=None, matlab_engine=None):
        W_true = DAG.scale_free(n_nodes=self.N_nodes, n_edges=self.N_edges)
        W_result_lst = []
        N_iterations = len(self.p["noise_scale"]) * len(range(self.N_res_samples))
        with tqdm(total=N_iterations) as pbar:
            for noise_scale in self.p["noise_scale"]:
                W_inner_result_lst = []
                for _ in range(self.N_res_samples):
                    data = IIDSimulation(W=W_true, n=self.N_learn_samples, method=self.method, sem_type=self.sem_type, noise_scale=float(noise_scale))
                    self.model.learn(data.X)
                    W_inner_result_lst.append((W_true - self.model.causal_matrix).flatten())
                    pbar.update(1)
                W_result_lst.append(W_inner_result_lst)
        return np.array(W_result_lst)


if __name__ == '__main__':
    
    logging.disable(logging.CRITICAL)

    N_nodes = 10
    N_edges = 20
    N_learn_samples = 1000
    N_res_samples = 100
    method = "linear"
    sem_type = "gauss"

    model = NoTearsLinearCausalDiscoveryModel(N_nodes, N_edges, N_learn_samples, N_res_samples, method, sem_type)
    model.set_parameters({"noise_scale": np.array([1, 10, 100])})
    results = model.simulate()

    print(results)

    np.save("NoTears_results.npy", results)
