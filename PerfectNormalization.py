#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:38:37 2024

@author: mali
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy,math

df=pd.read_csv('/Users/mali/Desktop/csv/multiple_linear_regression_dataset.csv')

train_set = df.to_numpy()

x=train_set[:,:2]
y=train_set[:,2]

x_features = ['Age','Experience']

fig,ax=plt.subplots(1, 2, figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(x[:,i],y)
    ax[i].set_xlabel(x_features[i])
ax[0].set_ylabel('Income')
plt.show()


def predict_model(x,w,b):
    
    f_wb=np.dot(x,w)+b
    
    return f_wb

w_init=[300,2000]
b_init=20000

x_vec = x[0,:]

pred1=predict_model(x_vec,w_init,b_init)

def compute_cost(x,y,w,b):
    m=x.shape[0]
    cost = 0 
    for i in range(m):
        f_wb = np.dot(x[i],w)+b
        cost += (f_wb - y[i])**2
    total_cost = cost/(2*m)
    return total_cost
    
first_cost = compute_cost(x,y,w_init,b_init)

def z_score_normalization(x):
    mu = np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    x_norm = (x-mu)/sigma
    
    return mu,sigma,x_norm

means,sigmas,x_norm=z_score_normalization(x)

x_mean = x-means

fig,ax=plt.subplots(1,3, figsize=(12,3),sharey=True)
ax[0].scatter(x[:,0],x[:,1])
ax[0].set_xlabel(x_features[0]);ax[0].set_ylabel(x_features[1])
ax[0].set_title('Unnormalized')
ax[0].axis('equal')



ax[1].scatter(x_mean[:,0],x_mean[:,1])
ax[1].set_xlabel(x_features[0]);ax[1].set_ylabel(x_features[1])
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')



ax[2].scatter(x_norm[:,0],x_norm[:,1])
ax[2].set_xlabel(x_features[0]);ax[2].set_ylabel(x_features[1])
ax[2].set_title('Normalized')
ax[2].axis('equal')
plt.show()

y_mu,y_means,y_norm = z_score_normalization(y)

def compute_gradient(x,y,w,b):
     m,n = x.shape
     dj_dw=np.zeros((n,))
     dj_db=0
     
     for i in range(m):
         f_wb = np.dot(x[i],w) + b
         for j in range(n):
             dj_dw[j] += (f_wb - y[i])*x[i,j]
         dj_db += f_wb - y[i]
         
     dj_dw = dj_dw / m
     dj_db = dj_db / m
     
     return dj_dw,dj_db

 
def gradient_descent(x,y,w_in,b_in,alpha,num_iter,gradient_function):
    
    w=copy.deepcopy(w_in)
    b=b_in
    
    for i in range(num_iter):
        dj_dw,dj_db=gradient_function(x,y,w,b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        return w,b

w_initialize = np.zeros_like(w_init)
b_initialize = 0
iterations = 1000
alpha = 1.0e-9

w_norm,b_norm = gradient_descent(x_norm,y,w_initialize,b_initialize,alpha,iterations,compute_gradient)

final_cost = compute_cost(x_norm,y_norm,w_norm,b_norm)












        
        
    
    


        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    