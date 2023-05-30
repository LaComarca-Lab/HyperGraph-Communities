import networkx as nx
import numpy as np
import xgi

def IRMM_algorithm(H, W=None, tol=1e-3):
    ''' Given a Hypergraph H, obtain the associated clusters via
    the Iterated Reweighted Modularity Maximization algorithm.
    '''

    # Define the associated weight and incidence matrices
    if not W:
        W = np.identity(len(H.edges))

    I = xgi.convert.to_incidence_matrix(H, sparse=True, index=False)
    
    # Define the delta_e list and D_v matrix
    delta_e = [len(edge) for edge in H.edges.members()]
    D_e = np.diag(delta_e)

    D_v = np.zeros((len(H.nodes),len(H.nodes)))
    for i, degree in enumerate(H.degree().values()):
        D_v[i,i] = degree

    # Loop updating the W matrix
    it = 0
    diff = np.inf
    while diff > tol:

        # Compute the reduced adjacency matrix of the hypergraph
        A = np.dot(I.dot(W), np.dot(np.linalg.inv((D_e - np.identity(len(H.edges)))), I.T.todense()))
        A -= np.diag(np.diag(A))

        # Apply the Louvain algorithm to the associated graph
        G = nx.from_numpy_array(A)
        clusters = nx.community.louvain_communities(G)

        # Loop over each edge e, updating the corresponding W[e,e]
        W_prev = np.copy(W)
        for e, edge in enumerate(H.edges.members()):
            
            # Intersect the edge's nodes with each cluster, computing the k_i's
            intersect_nodes = []
            for C in clusters:
                intersect_nodes.append(len(C.intersection(edge)))
            
            # Compute the new weight
            ksum = [1/(k_i+1) * (delta_e[e] + len(clusters)) for k_i in intersect_nodes] 
            wp = 1/len(edge) * np.sum(ksum)

            # Moving average with previous weight
            W[e,e] = 1/2 * (wp + W_prev[e,e])

        diff = np.abs(np.sum(W_prev - W))
        
        # Manual brake
        it += 1
        if it > 10000:
            raise Exception("No convergence after 10000 iterations")

    print(f'It converge after {it} iterations')

    return clusters    