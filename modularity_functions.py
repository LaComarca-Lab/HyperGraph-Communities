#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: Guillermo Vera, Jinling Yang, Gonzalo Contreras-Aso
Title: Functions related to the modularity and the algorithm defined in Kumar et. al.
'''

import networkx as nx
import numpy as np
import xgi
from scipy.cluster import hierarchy
from collections import defaultdict


def reduced_adjacency_matrix(H):
    """ Computes the reduced adjacency matrix of a hypergraph H,
    and the associated graph.
    """
    # Obtain the hypergraph incidence matrix
    W = np.identity(len(H.edges))
    I = xgi.convert.to_incidence_matrix(H, sparse=True, index=False)

    # Define the delta_e list and D_v matrix
    delta_e = [len(edge) for edge in H.edges.members()]
    D_e = np.diag(delta_e)

    # Compute the reduced adjacency matrix of the hypergraph
    A = np.dot(I.dot(W), np.dot(np.linalg.inv((D_e - np.identity(len(H.edges)))), I.T.todense()))
    A -= np.diag(np.diag(A))
    

    # Create the associated graph
    G = nx.from_numpy_array(A)

    # Relabeling the nodes to meet with H
    mapping = {}
    for cont, node in enumerate(H.nodes):
        mapping[cont] = node

    G = nx.relabel_nodes(G, mapping)
    
    return A, G, mapping


def calculate_modularities(H, G, Z, verbose=True):
    
    Modularity = []
    for n in range(len(H.nodes)):

        if n % 50 == 0 and verbose:
            print(f'-- {n/len(H.nodes)}% --')

        cuttree = hierarchy.cut_tree(Z, n_clusters = n)
        Communities_dict = {}

        for cont, i in enumerate(H.nodes):
            Communities_dict[i] = cuttree[cont][0]

        # Get the list of communities
        grouped_dict = defaultdict(list)
        for key, val in Communities_dict.items():
            grouped_dict[val].append(key)

        Communities_list = list(grouped_dict.values())

        q = nx.community.modularity(G, Communities_list)

        Modularity.append(q)
        
    return Modularity



def IRMM_algorithm(H, W=None, tol=1e-3, itmax = 10000, verbose = True):
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
        if it > itmax:
            raise Exception(f"No convergence after {itmax} iterations")

    if verbose:
        print(f'It converged after {it} iterations')

    return clusters    