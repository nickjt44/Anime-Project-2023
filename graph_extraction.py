import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
import time
import random

from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix, csc_matrix

from spektral.data import Graph
import pickle as pkl

class EnclosingSubgraph:
    """
    This class is for extracting the enclosing subgraph for a given user and anime rating pair. The enclosing
    subgraph captures relationships between the user-anime pair and its k-hop neighbors (k=1 for our purposes). The enclosing
    subgraphs will be fed in batches into the deep learning algorithm, which will learn the relationships represented
    by the enclosing subgraphs in order to predict the correct rating for each subgraph.
    
    Most of this code is adapted from Muhan Zhang's Github page.
    """
    def __init__(self,data):
        self.data = data
    
    def index_rows_cols(self):
        self.data_r = SparseRowIndexer(self.data)
        self.data_c = SparseColIndexer(self.data.tocsc())
    
    def append_new_user(self,train,test):
        self.user_train = train #np dense array of ratings for a single user
        self.user_test = test
        self.data = sparse.vstack([self.data,csr_matrix(self.user_train)])
        
    def extract_test_graphs(self,h,max_nodes):
        self.index_rows_cols()
        g_list = []
        i = self.data.tocoo().get_shape()[0] - 1
        for j in range(len(self.user_test)):
            r = self.user_test[j]
            if r == 0:
                continue
            print((i,j,r))
            temp = self.extract_graph(edge=(i,j),Arow=self.data_r,Acol=self.data_c,num_hops=h,g_label=r,max_nodes_per_hop=max_nodes)
            if temp == None:
                g_list.append([r])
            else:
                g_list.append(self.make_sp_graph(temp[0],temp[1],temp[2],temp[3],temp[4],temp[5]))
        return g_list
            
        
    def extract_graphs(self,h,max_nodes):
        self.index_rows_cols()
        g_list = []
        coo = self.data.tocoo()
        counter = 0
        for i, j, r in zip(coo.row,coo.col,coo.data):
            print(f"Count: {counter}")
            print((i,j,r))
            temp = self.extract_graph(edge=(i,j),Arow=self.data_r,Acol=self.data_c,num_hops=h,g_label=r,max_nodes_per_hop=max_nodes)
            if temp == None:
                g_list.append([r])
            else:
                g_list.append(self.make_sp_graph(temp[0],temp[1],temp[2],temp[3],temp[4],temp[5]))
            counter += 1
            if counter%10000 == 0:
                print(f"Dumping graph batch {counter/10000}")
                fname = 'graphs/train_graphs_' + str(counter)[0] + '.pkl'
                with open(fname, 'wb') as g:
                    pkl.dump(g_list,g)
                g_list = []
        return g_list
            
    def extract_graph(self,edge,Arow,Acol,num_hops,g_label,max_nodes_per_hop=None):
        """
        Extracts the k-hop (k represented by num_hops) subgraph for the user-anime pair represented by 'edge'.
        
        The algorithm for subgraph extraction can be referenced in Zhang's paper on the IGMC algorithm.
        
        This code is adapted from code from Muhan Zhang's Github page.

        """
        u_nodes, v_nodes = [edge[0]], [edge[1]]
        u_dist, v_dist = [0], [0]
        u_visited, v_visited = set([edge[0]]), set([edge[1]])
        u_fringe, v_fringe = set([edge[0]]), set([edge[1]])
        
        for dist in range(1,num_hops+1):
            v_fringe, u_fringe = self.neighbors(u_fringe, Arow), self.neighbors(v_fringe, Acol)
            u_fringe = u_fringe - u_visited
            v_fringe = v_fringe - v_visited
            u_visited = u_visited.union(u_fringe)
            v_visited = v_visited.union(v_fringe)
            
            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(u_fringe):
                    u_fringe = random.sample(u_fringe, max_nodes_per_hop)
                if max_nodes_per_hop < len(v_fringe):
                    v_fringe = random.sample(v_fringe, max_nodes_per_hop)
                    
            if len(u_fringe) == 0 and len(v_fringe) == 0:
                print('bla')
                break
            u_nodes = u_nodes + list(u_fringe)
            v_nodes = v_nodes + list(v_fringe)
            u_dist = u_dist + [dist] * len(u_fringe)
            v_dist = v_dist + [dist] * len(v_fringe)
    
        subgraph = Arow[u_nodes][:, v_nodes]
        
        subgraph[0, 0] = 0
        
        u, v, r = sparse.find(subgraph)  
        if (0 not in u) or (0 not in v):
            return None
        v += len(u_nodes)
        num_nodes = len(u_nodes) + len(v_nodes)
        node_labels = [x*2 for x in u_dist] + [x*2+1 for x in v_dist]
        max_node_label = 2*num_hops + 1
        
        return (u,v,r,node_labels,max_node_label,g_label)
        
        
    def neighbors(self,fringe, A):
        """
        Find all 1-hop neighbors of nodes in the fringe from A.
        """
        if not fringe:
            return set([])
        return set(A[list(fringe)].indices)
    
    def one_hot(self, idx, length):
      
        idx = np.array(idx)
        x = np.zeros([len(idx), length])
        x[np.arange(len(idx)), idx] = 1.0
        return x
        
    def make_sp_graph(self,u,v,r,node_labels,max_node_label,g_label):
        """
        Creates a Spektral Graph object. A Spektral Graph object consists of node labels, an adjacency matrix representing
        the graph structure, and the graph label. 
        
        In the case of the enclosing subgraph, the node labeling is the distance in hops of other nodes from the 
        target user and target item node. In the case that the number of hops is 1, the labeling will be 0 and 1 for the
        target user and anime, and 2 and 3 for neighbor users and anime.
        
        The adjacency matrix represents which nodes are connected and via what weight, and the graph label is
        the rating given by the target user for the target anime.
        
        """
        nodes = np.unique(np.concatenate((u,v),axis=None))
        labels_onehot = self.one_hot(node_labels,max_node_label+1)
        x = np.array(labels_onehot)

        adj = np.zeros((len(nodes),len(nodes)))
      
        for i in range(len(u)):
            adj[u[i],v[i]] = r[i]
            adj[v[i],u[i]] = r[i]
            
        y = np.array(g_label)
        a=csr_matrix(adj)
        
        return Graph(x=x,a=a,y=y)
    
class SparseRowIndexer:
    """
    This class is for easily indexing rows of sparse matrices. I did not write this class, it is
    from Muhan Zhang's Github page.
    """
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        indices = np.concatenate(self.indices[row_selector])
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        shape = [indptr.shape[0] - 1, self.shape[1]]
        return csr_matrix((data, indices, indptr), shape=shape)
    
class SparseColIndexer:
    """
    This class is for easily indexing columns of sparse matrices. I did not write this class, it is
    from Muhan Zhang's Github page.
    """
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []

        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csc_matrix.shape

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))

        shape = [self.shape[0], indptr.shape[0] - 1]
        return csc_matrix((data, indices, indptr), shape=shape)