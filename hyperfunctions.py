#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: Guillermo Vera, Jinling Yang, Gonzalo Contreras-Aso
Title: Functions related to the community calculation in hypergraphs
'''

import numpy as np
import networkx as nx
import statistics
from itertools import combinations, permutations
from collections import defaultdict
from scipy.cluster import hierarchy

import random
random.seed(10)


def adjacent_values_dict(H):
    '''his next function computes de adjacent values of a hypergraph from
    its nodes and hyperedges. It returns two dictionaries: 
    - degree_dict : it is a dictionary node : hyperdegree (a_{ii})
    - degree_i_j_dict: it is a dictionary node : hyperneighbors (a_{ij})
    
    Parameters
    ----------
    H : xgi.Hypergraph

    Returns
    -------
    hyperdegree_dict : dict
    aij_dict : dict
    '''

    # Get the hyperdegrees
    hyperdeg_dict = H.degree()

    aij_dict = defaultdict(lambda: 0) # dictionary with default value 0 for all possible entries

    for edge in H.edges.members():
        for i in edge:
            for j in edge:
                aij_dict[(i,j)] += 1
    return hyperdeg_dict, aij_dict



def derivatives_dict(hyperdeg_dict, aij_dict, verbose=False):
    '''Once we have the adjacency values of a hypergraph, we compute their
    derivative values. We set the infinity value as a 999.
    It returns a dictionary with edge (i,j) : dH/d{i,j}
    
    Parameters
    ----------
    hyperdeg_dict : dict
    aij_dict : dict
    verbose : bool, default False.

    Returns
    -------
    similar_dict : dict
    equivalent_nodes : list of tuples
    '''
    
    # Auxiliary function
    jaccard = lambda i, j: (hyperdeg_dict[i] + hyperdeg_dict[j] - 2 * aij_dict[(i, j)]) / aij_dict[(i, j)]

    # Initialize the returned variables
    similar_dict = {}
    equivalent_nodes = []

    # Iterate over each and every edge.
    for edge in aij_dict.keys():

        if aij_dict[edge] == 0: # If equivalent -> infinite derivative 

            equivalent_nodes.append(edge)
            if verbose:
                print(f'Nodes {edge[0]} and {edge[1]} are equivalent.')
            similar_dict[edge] = np.inf

        else:
            similar_dict[edge] = jaccard(*edge) # Compute the derivative

    return similar_dict, equivalent_nodes



def derivative_community_matrix(similar_dict, equivalent_nodes, threshold=None):
    '''Given a dictionary of the dH/d{i,j} per edge,
    create the associated "derivative" graph and compute
    from it the derivative adjacency matrix. If a threshold is
    given, return the community matrix too.
    
    Parameters
    ----------
    similar_dict : dict
    equivalent_nodes : dict
    threshold : None (default) or float or int

    Returns
    -------
    derivative_matrix : np.array
    community_matrix : np.array
    '''

    # Check the variable threshold
    assert not threshold or isinstance(threshold, float) or isinstance(threshold, int)

    # Create the adjacency graph
    G = nx.Graph()
    for (i,j), deriv in similar_dict.items():
        G.add_edge(i,j, weight=deriv)

    # Remove equivalent nodes from it
    for (i,j) in equivalent_nodes:
        G.remove_node(j) 

    # Sort nodes 
    Gsort = nx.Graph()
    Gsort.add_nodes_from(sorted(G.nodes(data=True)))
    Gsort.add_edges_from(G.edges(data=True))

    # Compute the derivative adjacency matrix
    derivative_matrix = nx.to_numpy_array(Gsort)

    if not threshold:
        return derivative_matrix 
    
    # Compute the community adjacency matrix (filter values above threshold)
    community_matrix = np.where(derivative_matrix  < threshold, derivative_matrix, 0)

    return derivative_matrix, community_matrix



def means_of_a_matrix(matrix):
    '''A function to help us to calculate the harmonic mean, normal mean and
    standard deviation from the derivative values (not considering infinity and 0 values)

    Parameters
    ----------
    matrix : np.array

    Returns
    -------
    harmonic_mean : float
    normal_mean : float
    des_tipica : float
    '''
    n = len(matrix)
    Har = []
    M = []

    for i in range(n):
        for j in range(n-i):
            # The first case we want to exclude 0 values to be able to
            # compute the harmonic mean, otherwise would make error dividing by 0
            if i <= j + i + 1 and matrix[i][j] > 0 and matrix[i][j] != np.inf:
                Har.append(matrix[i][j])
                M.append(matrix[i][j])

    harmonic_mean = statistics.harmonic_mean(Har)
    normal_mean = statistics.mean(M)
    des_tipica = statistics.stdev(Har)

    return harmonic_mean, normal_mean, des_tipica


def derivative_list(H, factor=10.0):
    """Given an XGI Hypergraph H, compute its derivative list. The
    factor parameter multiplies the maximum similarity for nodes not related.

    Parameters
    ----------
    H : xgi.Hypergraph
    factor : float

    Returns
    -------
    derivatives : list
    """

    # Compute the necessary matrices and dictionaries
    hyperdeg_dict, aij_dict = adjacent_values_dict(H)
    similar_dict, _ = derivatives_dict(hyperdeg_dict, aij_dict, verbose=False)
    max_similarity = np.max(list(similar_dict.values()))
    
    # Compute the derivate list
    derivatives = []     # It will contain all the derivatives in a list
    for (i,j) in combinations(list(H.nodes),2):
        if (i,j) in similar_dict.keys():
            derivatives.append(similar_dict[(i,j)])
        elif (j,i) in similar_dict.keys():
            derivatives.append(similar_dict[(j,i)])
        else:
            derivatives.append(factor*max_similarity) 

    return derivatives


def communities(H, derivative, method, threshold=None, n_clusters=None):
    """Given the derivative list and linkage method, perform 
    the clustering analysis and return the communities requested.

    Parameters
    ----------
    derivative : list
    method : string ("single", "complete", "average", "centroid", "ward")
    threshold : float
    n_clusters : int

    Returns
    -------
    communities_list : list
    """

    if not threshold and not n_clusters:
        raise Exception("Either a threshold or the number of clusters need to be provided.")

    # Create the linkage matrix
    Z = hierarchy.linkage(derivative, method)

    # Cut the linkage tree where specified    
    if threshold:
        cuttree = hierarchy.cut_tree(Z, height = threshold)
    elif n_clusters:
        cuttree = hierarchy.cut_tree(Z, n_clusters=n_clusters) 

    # Assign each node to its community
    node_community_dict = {}
    for index, node in enumerate(H.nodes):
        node_community_dict[node] = cuttree[index][0]

    # Assign each community its nodes
    communities_dict = defaultdict(set)
    for node, comm in node_community_dict.items():
        communities_dict[comm].add(node)

    return communities_dict
    
    
def height_based_cut(Z):
    """ Cut a dendrogram Z based at the highest height difference,
    return the height cut and the number of communities obtained.
    """
    
    # Calculate all heights
    Lamb = []
    for n in range(len(Z)):
        if n>=1:
            lamb = Z[n][2] - Z[n-1][2]
            Lamb.append(lamb)
            
    #print(Lamb)
            
    m = max(Lamb)
    num_fusion = [i for i, j in enumerate(Lamb) if j == m]
    index_max = num_fusion[0]
    #print(m, index_max)
    
    h_cut = (Z[index_max+1][2] + Z[index_max][2])/2
    
    return h_cut, num_fusion