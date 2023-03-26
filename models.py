# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 19:39:59 2022

@author: nickj
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from unidecode import unidecode
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from stellar_rgcn import RelationalGraphConvolution
from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix, csc_matrix

class MF_inc:
    """
    Implementation of Incremental Matrix Factorization in Tensorflow.
    The first three methods are vanilla Matrix Factorization, the last method (update) is
    the incremental step whereby the weights for the matrix are updated without having to
    run the entire model again.
    
    The code for this step is inspired by the paper Incremental learning for matrix factorization
    in recommender systems (2016) - Mengshoel et al.
    """
    def __init__(self,K,N,M,reg,mu):   
        """
        Initializes the parameters for Matrix Factorization.

        Parameters
        ----------
        K : Dimension of the embedding.
        N : Row-wise dimension of the input matrix.
        M : Column-wise dimension of the input matrix.
        reg : Regularization parameter.
        mu : Bias term.

        Returns
        -------
        None.

        """
        self.N = N
        self.M = M
        self.reg_U = reg[0]
        self.reg_A = reg[1]
        self.K = K
        self.mu = mu
        self.U = tf.keras.layers.Input(shape=(1,))
        self.A = tf.keras.layers.Input(shape=(1,))
        self.U_Embedding = tf.keras.layers.Embedding(self.N,self.K,embeddings_regularizer=tf.keras.regularizers.l2(self.reg_U),name="u_embed")(self.U)
        self.A_Embedding = tf.keras.layers.Embedding(self.M,self.K,embeddings_regularizer=tf.keras.regularizers.l2(self.reg_A),name="a_embed")(self.A)
        
    def create_model(self):
        """
        Creates the model to be used for Matrix Factorization. 

        """
        self.A_Bias = tf.keras.layers.Embedding(self.M,1)(self.A)
        self.U_Bias = tf.keras.layers.Embedding(self.N,1)(self.U)
        
        x = tf.keras.layers.Dot(axes=2)([self.U_Embedding,self.A_Embedding])
        x = tf.keras.layers.Add()([x,self.U_Bias,self.A_Bias])
        x = tf.keras.layers.Flatten()(x)
        
        self.model = tf.keras.models.Model(inputs=[self.U,self.A],outputs=x)
    
    def model_compile(self,lr=0.05):
        """
        Compiles the Matrix Factorization model with the learning rate as an input.

        """
        self.model.compile(loss='mse',
                          optimizer=tf.keras.optimizers.SGD(lr=lr,momentum=0.9),
                          metrics=['mse'])
        
    def model_fit(self,X,y,Xval,yval,epochs,batch_size):
        """
        Fits the model to the data.
        The input, output, validation input, validation output, number of epochs, and batch size are
        all used as inputs.

        """
        self.log = self.model.fit(x=X,y=y,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=(
                           Xval,
                           yval
                       )
                      )
    
    def update(self,data,reg):
        """
        The function for incremental updates of the Matrix Factorization model when a new user is added.
        The principle is that one can apply 'one sided least squares' to update the weight matrices
        since only new users are being added.
        
        The function takes the new data as input, in addition to a regularization term.
        """
        v = np.zeros(self.M)
        ratings = []
        indices = []
            
        for i in range(len(data)):
            aid = data[i][0]
            rating = data[i][1]
            indices.append(aid)
            ratings.append(rating - self.mu)
        
        v = np.array(ratings)

        X = self.model.layers[3].get_weights()[0][indices,:]
            
        b = self.model.layers[6].get_weights()[0][indices]
        v = v - b.T
        v = v.T
            
        new_user_weights = np.random.rand(self.K)
        new_user_bias = np.sum(v)/(len(data))
        
        mat = X.T.dot(X) + reg*np.eye(self.K)
        vector = X.T.dot(v-new_user_bias)
        
        wts = np.linalg.solve(mat,vector)
            
        new_user_weights = wts
        temp = [new_user_weights.T.dot(self.model.layers[3].get_weights()[0][i]) for i in indices]
        new_user_bias = np.sum(v-temp)/len(data)
            
        recommendations = {}
        for i in range(self.M):
            if i in indices:
                continue
            else:
                rec = new_user_weights.T.dot(self.model.layers[3].get_weights()[0][i]) + self.mu + new_user_bias + self.model.layers[6].get_weights()[0][i]
                recommendations[i] = rec
        return recommendations
    

class IGMC(tf.keras.Model):
    """
    This class implements the IGMC matrix completion algorithm.
    
    This algorithm has been written in Tensorflow by me, based on a paper by Muhan Zhang and 
    Yixin Chen.
    
    The algorithm is specifically for the dataset I am using ie. MyAnimeList user/anime scores.
    
    The data will be fed into the algorithm in the form of graph objects called the enclosing subgraphs of
    a user-rating pair.
    
    The enclosing subgraph of a user-rating pair is a way of encoding the relationship between a specific user
    and anime series via their k-hop neighbors (k being an adjustable parameter set to 1 for this project).
    
    """
    def __init__(self,n_relations=10,n_bases=2,latent_dim=[32,32,32,32]):
        """
        Parameters
        ----------
        n_relations : TYPE int
            DESCRIPTION. The number of possible scores a user can give a specific show, the variable
            we wish to predict.
        n_bases : TYPE int
            DESCRIPTION. Parameter for basis decomposition. The default is 2.
        latent_dim : TYPE list[int]
            DESCRIPTION. Dimension of the convolutional layers. Default is [32,32,32,32].

        Returns
        -------
        None.
        
        Description
        -----------
        Initializes the convolutional layers using the RelationalGraphConvolution object. Also initializes
        the Dense layers.

        """
        super().__init__()
        self.convs = []
        self.convs.append(RelationalGraphConvolution(units=latent_dim[0], 
                                                  num_relationships=n_relations, num_bases=n_bases))
        for i in range(0,len(latent_dim)-1):
            self.convs.append(RelationalGraphConvolution(input_dim=latent_dim[i],units=latent_dim[i+1], 
                                                      num_relationships=n_relations, num_bases=n_bases))
        self.dense1 = Dense(128)
        self.dense2 = Dense(1)
        
    def call(self,data):
        """
        Implements the IGMC deep graph learning algorithm in Tensorflow.
        
        The input data is a batch of enclosing subgraphs in disjoint form, each subgraph representing a user-anime pair.
        
        First, the adjacency matrix representing the graph structure (with each nonzero value equal to the rating a given
        user gives a given anime) is decomposed into a list of normalized adjacency matrices for each rating (necessary
        input for the convolutional layers).
                                                                                                              
        The convolutional layers are then applied to the list of adjacency matrices with tanh activation functions, and the outputs are stored in
        the concat_states variable.
        
        The next step is to pool the features from the convolutional layer process. This is done via concatenating the 
        vector representations of the target user and item that are output from the convolution process, as outlined
        in Zhang's paper.
        
        Finally, the dense layers transform this representation into the output.
        """
        
        x, a = data[0],data[1]
        
        adj_list = []
        a = to_sparse_tensor(a)
        a_dense = tf.sparse.to_dense(a)

        for i in range(1,11):
            adj_list.append(tf.sparse.from_dense(normalizeAdjacency(tf.where((a_dense == i).numpy(),a_dense,0).numpy()/i)))
           
            #adj_list.append(to_sparse_tensor(normalizeAdjacency(np.where((a_dense == i),a_dense,0)/i)))
            
        concat_states = []
        x = tf.expand_dims(x,axis=0)
        for conv in self.convs:
            y = [x]
            y.extend(adj_list)
            x = conv(y)
            x = tf.keras.activations.tanh(x)
            concat_states.append(x)
        concat_states = tf.keras.layers.Concatenate(axis=2)(concat_states)
         
        users = data[0][:, 0] == 1
        items = data[0][:, 1] == 1
        
        concat_states = tf.squeeze(concat_states)
        
        x = tf.keras.layers.Concatenate(axis=1)([concat_states[users], concat_states[items]])
        x = self.dense1(x)
        x = tf.keras.activations.relu(x)
        x = self.dense2(x)
        return x

def normalizeAdjacency(W):
    """
    Normalizes the input adjacency matrix and returns it.

    """
    assert W.shape[0] == W.shape[1]
    x = W.sum(axis=1)
    W = W / (x[:,np.newaxis] + 1)
    return (W)

def to_sparse_tensor(matrix):
    """
    Converts dense matrix or tensor to sparse tensor.

    """
    if tf.is_tensor(matrix):
        return(tf.sparse.from_dense(matrix))
    matrix = matrix.tocoo()
    inds = np.mat([matrix.row, matrix.col]).transpose()
    return tf.SparseTensor(inds, matrix.data, matrix.shape)