#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:14:47 2024

@author: mali
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df=pd.read_csv('/Users/mali/Downloads/multiple_linear_regression_dataset (1).csv',sep=';')
x=df.iloc[:,:2]
y=df.iloc[:,2]
x_train=x.to_numpy()
y_train=y.to_numpy()

x_features = ['Deneyim','Yaş']

w_init=[30,0.3]
b_init=759.6412

fig,ax = plt.subplots(1, 2 , figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(x_train[:,i],y_train)
    ax[i].set_xlabel(x_features[i])
ax[0].set_ylabel('Maaş')
plt.show()

i=2


def compute_cost_function(x,y,w,b):
    """
     

     Parameters
     ----------
     x : (ndarray) shape (m, n) type np.ndarray.
     y : (ndarray) shape (m,) type np.ndarray.
     w : (ndarray) shape (n,) type np.ndarray.
     b : (scalar) type int.

     Returns
     -------
     total_cost : (scalar) type int.

     """ 
    m=x.shape[0]
    cost = 0 
    for i in range(m):
        f_wb = np.dot(x[i],w)+b
        cost += (f_wb -y[i])**2
    total_cost = cost/(2*m)
    return total_cost
cost_first = compute_cost_function(x_train,y_train,w_init,b_init)

print(f'cost value {cost_first}') 
## first cost value is 7927504.582454435

#%% Normalization (z-score normalization)
def z_score_normalization(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_norm = (x-mu)/sigma
    
    return x_norm,sigma,mu
    
x_norm,sigma,mu=z_score_normalization(x_train)

cost_second=compute_cost_function(x_norm,y_train,w_init,b_init)

print(f'cost second {cost_second}')
## second cost value is 268925.14157785464
## Even without implementing gradual reductions, we have reduced our costs significantly.




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    