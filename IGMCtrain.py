# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 09:23:34 2022

@author: nickj
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from unidecode import unidecode
import time
import random
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Dense, Dropout
import pickle as pkl

from stellar_rgcn import RelationalGraphConvolution

from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix, csc_matrix

from spektral.data import Dataset, DisjointLoader, Graph, BatchLoader
from spektral.data.utils import to_disjoint

from models import IGMC

from graph_extraction import EnclosingSubgraph

tf.config.run_functions_eagerly(True)

class CustomGenerator(tf.keras.utils.Sequence):
    """
    This class generates the data for the deep learning model to learn from.
    
    The data is either for training or validation, determined by one of the arguments to init. The 
    methods to construct the enclosing subgraphs are called for each batch of data, and these graphs are fed
    into the machine learning algorithm in disjoint form (ie. as one large matrix, since this is how some of the underlying
    layers best handle the data).
    
    I limited the batch size to 8, since in disjoint form a lot of memory is used, and I didn't want to overload my
    laptop.
    """
    def __init__(self,data,anime,csr,batch_size=8,data_type='train',split=0.9):
        self.batch_size = batch_size
        if data_type == 'train':
            self.data = data.iloc[:int(len(data)*split),:]
        elif data_type == 'val':
            self.data = data.iloc[int(len(data)*split):,:]
            
        self.anime = anime
        self.csr = csr
        self.sg = EnclosingSubgraph(csr)
        self.sg.index_rows_cols()
        self.valid = self.data.shape[0]
        
    def __len__(self):
        l = int(np.floor(self.valid/self.batch_size))
        return l
    
    def __getitem__(self,idx):
        idxs = [item for item in range(idx*self.batch_size,(idx+1)*self.batch_size)] 
        g_list = []
        x_list = []
        a_list = []
        t_list = []
        for i,j in zip(self.data.UserID.iloc[idxs],self.data.AnimeID.iloc[idxs]):
            r = self.csr[i,j]
            temp = self.sg.extract_graph(edge=(i,j),Arow=self.sg.data_r,Acol=self.sg.data_c,num_hops=1,g_label=r,max_nodes_per_hop=600)
            if temp != None:
                g = self.sg.make_sp_graph(temp[0],temp[1],temp[2],temp[3],temp[4],temp[5])
                #g_list.append([g.x,to_sparse_tensor(g.a)])
                x_list.append(g.x)
                a_list.append(g.a)
                t_list.append(g.y)
        if len(t_list) == 0:
            return None
        return [to_disjoint(x_list,a_list),t_list]
    
    


class MyDataSet(Dataset):
    """
    Class that inherits from the spektral Dataset class in order to handle the data easily.
    """
    def __init__(self,data,**kwargs):
        self.data = data
        super().__init__(**kwargs)
        
    def read(self):
        return self.data
    
def ARRNorm(strength,coeffs):
    """
    This function limits the difference in weights for adjacent ratings by applying a penalty.  It's controlled
    by a strength parameter.
    """
    reg = 0
    for i in range(len(coeffs)-1):
        reg += tf.norm((coeffs[i] - coeffs[i+1]))**2
    return strength*reg



def val_step(inputs,target):
    """
    This function computes a validation step via calling the model on the input batch, and comparing the predictions
    to the target value. "training=False" is an input to the model since we don't want the model to learn from validation
    data.

    """
    predictions = model(inputs,training=False)
    loss = loss_fn(target,tf.transpose(predictions))
    return loss


def train_step(inputs,target,step):
    """
    This function computes a training step via calling the model on the input batch, with training set to True,
    and comparing the predictions to the target value, in addition to the regularization loss computed by
    the adjacent regularization term, which limits the magnitude of the difference between weight values corresponding
    to adjacent ratings (eg. 5,6 or 9,10).
    
    Tensorflow's GradientTape class is used to implement the training and feeding the gradients back into the optimizer
    to adjust the weights.

    """
    tf.config.run_functions_eagerly(True)
    if step%200 == 0:
        print(target)
    with tf.GradientTape() as tape:
        predictions = model(inputs,training=True)
        reg_loss = 0
        for conv in model.convs:
            coeffs = conv.r_kernels
            reg_loss += ARRNorm(0.00001,coeffs)
        loss = loss_fn(target,tf.transpose(predictions))
        total_loss = loss + reg_loss
    if step%200 == 0:
        print(predictions)
        print(f"Loss: {loss}, reg loss: {reg_loss}")
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# The following lines of code comprise the training process of the IGMC algorithm.

data = pd.read_csv('users_for_training_new_mapping.csv',index_col=0)
anime = pd.read_csv('anime_REINDEXED.csv',index_col=0)
data = data[data.User_Mapped.isin(shuffle(data.User_Mapped.unique(),random_state=42)[:6000])]
unq = data.User_Mapped.unique()

for i in range(len(unq)):
    print(i)
    data.UserID[data.User_Mapped == unq[i]] = i

shape = (data['UserID'].max()+1,len(anime))
coo = coo_matrix((data["Score"], (data["UserID"], data["AnimeID"])), shape=shape)
csr = coo.tocsr()

data = data.sample(frac=1,random_state=42)

tf.config.run_functions_eagerly(True)
model = IGMC()
#model.run_functions_eagerly=True
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0025)
train_gen = CustomGenerator(data=data,anime=anime,csr=csr,batch_size=4,data_type='train')
val_gen = CustomGenerator(data=data,anime=anime,csr=csr,batch_size=1,data_type='val')
l = train_gen.__len__()
epochs = 20
epoch = 1
step = 1
losses = []
total_losses = []
val_losses = []
total_val_losses = []

tf.test.is_gpu_available()

#model.built=True
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
flag = 0
while epoch < epochs+1:
    #print('ha')
    for batch in train_gen.__iter__():
        if batch == None:
            print("Empty Batch")
            step += 1
            continue
        if step == 2:
            print("Loading Weights...")
            model.load_weights('checkpoints/model_1m/epoch_' + str(5) + '.h5')
            # for i,w in enumerate(model.coefficients):
            #     print(i,w.name,w)
            for i,conv in enumerate(model.convs):
                print(i)
                coeffs = []
                for coeff in conv.coefficients:
                    coeffs.append(coeff.numpy())
                print(coeffs)
            #model.save('checkpoints/saved_model')
            
                
                
        if step%200 == 0:
            for i,conv in enumerate(model.convs):
                print(i)
                coeffs = []
                for coeff in conv.coefficients:
                    coeffs.append(coeff.numpy())
                print(coeffs)
            
            print(f"Epoch {epoch} of {epochs}")
            print(f"Step {step} of {l}")
            print(f"RMSE: {np.sqrt(np.mean(losses))}")
            losses = []
            #model.save_weights('checkpoints/model_1m/epoch_' + str(epoch) + '.h5')
        
        loss = train_step(*batch,step)
        losses.append(loss)
        step += 1
    #model.save_weights('checkpoints/model_1m/epoch_' + str(epoch) + '.h5')
    epoch += 1
    print("===== Validation =====")
    losses = []
    for batch in val_gen.__iter__():
        if batch == None:
            print("Empty Batch")
            continue
        val_loss = val_step(*batch)
        val_losses.append(val_loss)
        if len(val_losses)%500 == 0:
            print(len(val_losses))
            print(np.sqrt(np.mean(val_losses)))
    total_val_losses.append(np.sqrt(np.mean(val_losses)))
    print(f"Avg Validation Losses: {total_val_losses}")
    val_losses = []
    step = 1