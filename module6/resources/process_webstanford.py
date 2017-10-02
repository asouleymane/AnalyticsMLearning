import os, sys
import itertools
import pickle
import numpy as np
from collections import defaultdict

def create_graph(threshold, transition_matrix=False):
    """ Create graph in adjacency list and transition matrix
    threshold: used to threshold out a subgraph to reduce memory consumption """
    
    # Graph in adjacency list, implemented as defaultdict of sets
    G = defaultdict(set)
    
    if not os.path.exists('/tmp/web-Stanford.{}.pkl'.format(threshold)):
        with open('/tmp/web-Stanford.txt') as f_edges:
            while 1:
                line = f_edges.readline()
                if not line:
                    break
                if line.startswith('#'):
                    continue
                fro, to = list(map(int, line.split()))
                if fro<=threshold and to<=threshold:
                    G[fro].add(to)
            with open('/tmp/web-Stanford.{}.pkl'.format(threshold), 'wb') as f_adj:
                pickle.dump(G, f_adj)

    # Size of graph
    num_vertices = 0
    num_edges = 0
    with open('/tmp/web-Stanford.{}.pkl'.format(threshold), 'rb') as f_adj:
        G = pickle.load(f_adj)
        for k, v in G.items():
            num_edges += 1
            num_vertices = max(num_vertices, max(k, max(v)))

    print('Graph size', 'V={} E={}'.format(num_vertices, num_edges))

    if transition_matrix:
        if not os.path.exists('/tmp/web-Stanford.{}.npy'.format(threshold)):
            # Connectivity matrix
            matG = np.zeros((num_vertices, num_vertices), dtype=bool)
            for u, V in G.items():
                for v in V:
                    matG[u-1, v-1] = 1

            # Compute transition matrix
            out_degree = np.sum(matG, axis=1)
            nonzero = np.flatnonzero(out_degree)
            matP = matG.copy().astype(float)
            matP[nonzero, :] /= out_degree[nonzero][..., np.newaxis]
            np.save('/tmp/web-Stanford.{}.npy'.format(threshold), matP)
        else:
            matP = np.load('/tmp/web-Stanford.{}.npy'.format(threshold))

        print('Transition matrix shape', matP.shape)

if __name__=='__main__':
    create_graph(16000)
    create_graph(5600)
