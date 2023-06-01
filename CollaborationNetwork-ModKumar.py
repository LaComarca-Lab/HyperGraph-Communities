#!/usr/bin/env python
# coding: utf-8

# # Clustering Stefano Boccaletti's collaboration network


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import xgi
from collections import defaultdict
from itertools import combinations
import pandas as pd

from hyperfunctions import *
from modularity_functions import *

import time


MEAN = 10


# ##  Constructing the Hypergraph

# We create the hypergraph with Stefano's coauthors and their respective coauthors.

df = pd.read_csv('Datasets/boccaletti_and_cocoauthors.csv', sep = ';')

hyperedge_dict = {}
for paper, ID in enumerate(df['Author(s) ID']):
    hyperedge_dict[paper] = ID.split(';')[:-1]

H = xgi.Hypergraph(hyperedge_dict)


# We create the hypergraph with only Stefano's coauthors. We can apply some restrictions to the number of papers with him to be considered a coauthor.

restriction_papers = 3

dg = pd.read_csv('Datasets/boccaletti_coauthors.csv', sep = ',')

hyperedge_dic = {}
for paper, ID in enumerate(dg['Author(s) ID']):
    hyperedge_dic[paper] = ID.split(';')[:-1]

E = xgi.Hypergraph(hyperedge_dic)

for node in list(E.nodes):
    if E.degree(node) < restriction_papers:
        E.remove_node(node)     


# Keep from the first hypergraph (H) the nodes from the second hypergraph (E)


for node in list(H.nodes):
    if node in list(E.nodes):
        continue
    else:
        H.remove_node(node)
        
# Also remove Stefano Boccaletti
H.remove_node('7006291912')

H.remove_edges_from(H.edges.singletons())



# In[7]:


nodes_list = list(H.nodes)


# In[8]:


Authors_list = []
IDs_list = []
for author, ID in zip(dg['Authors'],dg['Author(s) ID']):
    Authors_list.append(author.split(', '))
    IDs_list.append(ID.split(';')[:-1])

authors_dict = {}
IDs_dict={}
for cont, authors in enumerate(Authors_list):
    for cont_2, author in enumerate(authors):
        IDs_dict[IDs_list[cont][cont_2]] = author
        authors_dict[author] = IDs_list[cont][cont_2]
        


# ## Communities

    
# ## Modularity (Kumar)

# In[ ]:


time_modularity_Kumar = [] 

for it in range(MEAN):
    
    print(f"Kumar iteration {it}")
    
    time_mod_K_i = time.time()

    clusters = IRMM_algorithm(H, tol=1e-3, verbose=False)
    
    time_mod_K_f = time.time()

    time_modularity_Kumar.append(time_mod_K_f - time_mod_K_i)
    
    
    
print("Modularity (IRMM) - Average time:", np.mean(time_modularity_Kumar))


# *Conclusion*: the general method coincides with our modularity maximization in this example

# In[ ]:
    
A, G, mapping = reduced_adjacency_matrix(H)
    
Communities_K = []
for comm in clusters:
    Community_K = []
    Community_IDs = []
    for node in comm:
        Community_K.append(mapping[node])
        Community_IDs.append(IDs_dict[mapping[node]])
    Communities_K.append(Community_K)

q = nx.community.modularity(G, Communities_K)
print(f'Number of communities (IRMM): {len(Communities_K)},')
print(f'The modularity for this partition is: {q}')


# In[ ]:




