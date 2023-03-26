# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:38:39 2023

@author: nickj
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from unidecode import unidecode
import time
import heapq
from tensorflow import keras

from models import MF_inc


df = pd.read_csv('users_for_training_new_mapping.csv',index_col=0)
anime = pd.read_csv('anime_REINDEXED.csv',index_col=0)
data = df[df.User_Mapped.isin(shuffle(df.User_Mapped.unique(),random_state=42)[:5000])]
calibration_data = df[df.User_Mapped.isin(shuffle(df.User_Mapped.unique(),random_state=42)[5000:6000])]

user_mapping = {v: k-1 for k, v in enumerate(data['UserID'].unique(), 1)}
data['User_Mapped_MF'] = data.UserID.replace(user_mapping)
data = data.sample(frac=1,random_state=42)
unq = data.User_Mapped_MF.unique()

train_data = data.iloc[:int(len(data)*0.9),:]
val_data = data.iloc[int(len(data)*0.9):,:]

N = len(data['User_Mapped_MF'].unique())
M = anime.shape[0]
regs = [1,0.5,0.25,0.1,0.01,0.001,0.0001,0]
lrs = [0.1,0.05,0.025,0.01]
mu = train_data.Score.mean()
model_reg = [0,0]

print(train_data['AnimeID'].head())
print(val_data.User_Mapped_MF.unique())
print(N)
print(M)

def test_learning_rate():
    """
    This function determines the ideal learning rate for the Matrix Factorization model via iterating
    through the lrs list. The learning rate with the lowest validation loss after 10 epochs will be selected.

    """
    results = []
    for lr in lrs:
        mf = MF_inc(20,N,M,model_reg,mu)
        mf.create_model()
        mf.model_compile(lr)
        mf.model_fit(X=[train_data['User_Mapped_MF'],train_data['AnimeID']],y=train_data['Score'] - mu,
                    Xval=[val_data['User_Mapped_MF'],val_data['AnimeID']],
                    yval=val_data['Score'] - mu,
                    epochs=10,batch_size=128)
        
        log = mf.log
        
        results.append((lr,log.history['val_loss']))
        print(results)
    
c_mapping = {v: k+5000-1 for k, v in enumerate(calibration_data['UserID'].unique(), 1)}

def calibration_data(cdata=calibration_data):  
    """
    This function is used to generate data used to calibrate the update method of the matrix factorization model.
    The data will be separated into 'known' and 'unknown' subsets, as will the final test data, and will be fed into
    the update method to determine the best regularization term.

    """
    cdata['User_Mapped_MF'] = cdata.UserID.replace(c_mapping)
    vc = cdata.UserID.value_counts()
    cdata['vc'] = cdata.apply(lambda x: vc[x[0]],axis=1)
    cdata = cdata[cdata.vc >= 100]
    cdata = cdata.drop('vc',axis=1)
    
    print(len(cdata.UserID.unique()))
    
    for user in cdata.User_Mapped_MF.unique():
        userdata = cdata[cdata['User_Mapped_MF'] == user]
        if len(userdata) <= 200:
            cutoff = int(len(userdata)/2)
        else:
            cutoff = 100
        userdata.iloc[:cutoff,:].to_csv('calibration_known/User_'+str(user)+'.csv')
        userdata.iloc[cutoff:,:].to_csv('calibration_unknown/User_'+str(user)+'.csv')

def relevance(score,zscore):
    if score >= 9:
        return 1
    elif zscore >= 1:
        return 1
    else:
        return 0

def calibrate_model(lr):
    """
    This function runs the MF model at the optimal learning rate, and then implements the update function on the
    calibration data to find the ideal regularization term.

    """
    count = 0
    error = 0
    rmses = []
    MAPs = []
    model = MF_inc(20,N,M,model_reg,mu)
    model.create_model()
    model.model_compile(lr)
    model.model_fit(X=[train_data['User_Mapped_MF'],train_data['AnimeID']],y=train_data['Score'] - mu,
                Xval=[val_data['User_Mapped_MF'],val_data['AnimeID']],
                yval=val_data['Score'] - mu,
                epochs=10,batch_size=128)
    for reg in regs:
        count = 0
        sqerror = 0
        users = 0
        APsum = 0
        MAP = 0
        for f in os.listdir('calibration_known'):
            users += 1
            known = pd.read_csv('calibration_known/' + f)
            unknown = pd.read_csv('calibration_unknown/' + f)
            zmean = known['Score'].mean()
            zstd = known['Score'].std()
            unknown['Zscore'] = (unknown['Score'] - zmean)/zstd
            known_data = [(known.loc[i,'AnimeID'],known.loc[i,'Score']) for i in known.index]
            unknown_data = [(unknown.loc[i,'AnimeID'],unknown.loc[i,'Score']) for i in unknown.index]
            recs = model.update(data=known_data,reg=reg)
            unknown['Prediction'] = unknown['AnimeID'].apply(lambda x: recs[x][0])
            print(unknown['Prediction'])
            largest = unknown.nlargest(5,'Prediction')
            
            # calculate mse
            for d in unknown_data:
                try:
                    sqerror += (recs[d[0]]-d[1])**2
                    count += 1
                except:
                    continue
                
            # calculate MAP
            rel = 0
            AP = 0
            for i in range(len(largest.index)):
                rel += relevance(largest.loc[largest.index[i],'Score'],largest.loc[largest.index[i],'Zscore'])
                AP += (rel/(i+1))
            AP = AP/len(largest)
            
            APsum += AP
            MAP = APsum/users
                
            print(sqerror)
            print(count)
            print(MAP)
        rmse = np.sqrt(sqerror/count)
        rmses.append((reg,rmse))
        
        MAPs.append((reg,MAP))
        
        print("Final RMSEs: ")
        print(rmses)
        
        print("Final MAPs: ")
        print(MAPs)
   
def train_final_model(lr=0.05):
    """
    This function trains the final model on all training data (6000 users) to be used for evaluation and
    comparison with the IGMC model.
    """
    df = pd.read_csv('users_for_training_new_mapping.csv',index_col=0)
    anime = pd.read_csv('anime_REINDEXED.csv',index_col=0)
    data = df[df.User_Mapped.isin(shuffle(df.User_Mapped.unique(),random_state=42)[:6000])]
    
    user_mapping = {v: k-1 for k, v in enumerate(data['UserID'].unique(), 1)}
    data['User_Mapped_MF'] = data.UserID.replace(user_mapping)
    data = data.sample(frac=1,random_state=42)

    train_data = data.iloc[:int(len(data)*0.9),:]
    val_data = data.iloc[int(len(data)*0.9):,:]

    N = len(data['User_Mapped_MF'].unique())
    M = anime.shape[0]
    mu = train_data.Score.mean()
    model_reg = [0,0]
    
    mf = MF_inc(20,N,M,model_reg,mu)
    mf.create_model()
    mf.model_compile(lr)
    mf.model_fit(X=[train_data['User_Mapped_MF'],train_data['AnimeID']],y=train_data['Score'] - mu,
                Xval=[val_data['User_Mapped_MF'],val_data['AnimeID']],
                yval=val_data['Score'] - mu,
                epochs=10,batch_size=128)
    mf.model.save_weights('checkpoints/model_mf' + '.h5')
    
#test_learning_rate()
#calibration_data()
#calibrate_model(0.05)

train_final_model()
            
