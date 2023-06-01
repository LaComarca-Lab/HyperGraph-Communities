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


time_derivative = []

for it in range(MEAN):
    
    time_deriv_i = time.time()
    
    derivative = derivative_list(H)
    
    time_deriv_f = time.time()
    
    time_derivative.append(time_deriv_f - time_deriv_i)

# In[10]:

print("Derivative - Average time:", np.mean(time_derivative))


method = "average"

# In[11]:



time_dendrogram = [] 

for it in range(MEAN):    
    time_dendro_i = time.time()
    
    Z = hierarchy.linkage(derivative, method)
    
    time_dendro_f = time.time()
    
    time_dendrogram.append(time_dendro_f - time_dendro_i)
    

print("Dendrogram - Average time:", np.mean(time_dendrogram))



# ## Modularity (us)

# In[ ]:


time_modularity_us = [] 

for it in range(MEAN):
    
    time_mod_us_i = time.time()

    A, G, mapping = reduced_adjacency_matrix(H)

    Modularity = calculate_modularities(H, G, Z, verbose=False)

    m = max(Modularity)
    num_fusion = [i for i, j in enumerate(Modularity) if j == m]
    index_max = num_fusion[0]
    
    
    communities_dict = communities(H, derivative, method, n_clusters=index_max)
    
    time_mod_us_f = time.time()
    
    time_modularity_us.append(time_mod_us_f - time_mod_us_i)
    
# In[ ]:


print("Modularity Us - Average time:", np.mean(time_modularity_us))


# *Conclusion*: the general method coincides with our modularity maximization in this example

# In[ ]:


print(f'Number of communities maximizing the modularity is: {index_max},')
print(f'with modularity (us): {m}')



