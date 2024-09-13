#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 14:35:39 2021

@author: navid
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


import matplotlib as mpl

mpl.rcParams['figure.titlesize'] = 25 
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.labelsize'] = 22 
mpl.rcParams['xtick.labelsize'] = 22 
mpl.rcParams['ytick.labelsize'] = 22 
mpl.rcParams['legend.fontsize'] = 18


np.random.seed(1)

class RESERVOIR():

    def __init__(self, data , n_reservoir=200, n_inputs=1 , n_outputs=1 ,
                 spectral_radius = 0.95, sparsity=0 , learning_rate=0.2):
        """
        Args:
            n_reservoir: nr of reservoir neurons
            spectral_radius: spectral radius of the recurrent weight matrix
            sparsity: proportion of recurrent weights set to zero
        """
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.data = data
        self.test_data = data[int(0.8*len(data)):]
        self.learning_rate = learning_rate
        
    def initweights(self):
        # initialize reservoir weights:
        # begin with a random matrix centered around zero:
        W = np.random.rand(self.n_reservoir, self.n_reservoir)*2-1
        W = W / (np.linalg.norm(W)+0.001)    
        # delete the fraction of connections given by (self.sparsity):
        for i in range(self.n_reservoir):
            for j in range(self.n_reservoir):
                if np.random.uniform() > self.sparsity:
                    W[i,j] = 0
                            
        # compute the spectral radius of these weights:
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        # rescale them to reach the requested spectral radius:
        self.W = W * (self.spectral_radius / radius)

        # random input weights:
        self.W_in  = np.random.rand(self.n_reservoir , self.n_inputs )*0.2 - 0.1
        self.W_in = self.W_in/(np.linalg.norm(self.W_in)+0.001)
        # random output weights:
        self.W_out = np.random.rand(self.n_reservoir , self.n_outputs)*0.2 - 0.1
        self.W_out = self.W_out/(np.linalg.norm(self.W_out)+0.001)
        # initializing the neurons with ranodm values
        self.state = np.random.zeros((self.n_reservoir , 1))
        self.R = self.state
                              
    def _update(self, target_pattern, output_pattern):
        """Updates the output weights"""
        self.W_out += self.learning_rate*(target_pattern - output_pattern)*(self.state)
        
    def _update_state(self , input_pattern):
        """Calculates the output pattern"""
        self.state = np.tanh(np.dot(self.W, self.state) + np.dot(self.W_in, input_pattern))
        return np.dot(np.transpose(self.W_out) , self.state)[0,0]


    def _train(self):
        self.clf = Ridge(alpha=0.1)
        
        for train in range(500):
            self.R = np.zeros((2000,self.n_reservoir))
            traj = [self.data[0]]
            for i in range(2000):
                self.R[i] = self.state.reshape(1,self.n_reservoir)
                traj.append(self._update_state(traj[-1]))
                

            self.clf.fit(self.R, self.data[:2000])
            if train%100==0:
                print(f'training {train} is doen!')
                #print(np.sum(np.abs(self.clf.coef_.reshape(self.n_reservoir , 1) - self.W_out)))
            self.W_out = self.clf.coef_.reshape(self.n_reservoir , 1)

            
        plt.figure(figsize=(22,9))
        plt.plot(traj, label='train')
        plt.plot(self.data[:2000] , label='goal')
        plt.legend()
        plt.show()
            
            
            
        
    def _predict(self, input_pattern):
        self.state = np.tanh(np.dot(self.W, self.state) + np.dot(self.W_in, input_pattern))
        return np.dot(np.transpose(self.W_out),self.state)

        
    def _test(self):
        x = self.test_data[0]
        self.prediction = [x]
        for i in range(int(0.2*len(self.data))-1):
            self.prediction.append(self._predict(self.prediction[-1]))
    
    def _plot(self):
        landa1 = 0.906
        T = np.linspace(0,10,500)*landa1
        plt.figure(figsize=(20,9))
        plt.plot(T , self.prediction , label='Prediction')
        plt.plot(T , self.test_data , label='Actual')
        plt.plot(np.ones(15)*landa1 , np.linspace(-30,30,15) , 'r--' , label='Lyaponov time')
        plt.legend()
        plt.ylim(-30,30)
        plt.xlabel(r'$X_2(t)$')
        plt.ylabel(r'$\lambda_1 t$')
        plt.savefig('comparison.png')
    
    def run(self):
        self.initweights()
        self._train()
        self._test()
        self._plot()
        