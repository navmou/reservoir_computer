#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:55:32 2021

@author: navid
"""


import numpy as np
import matplotlib.pyplot as plt


import matplotlib as mpl

mpl.rcParams['figure.titlesize'] = 25 
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.labelsize'] = 22 
mpl.rcParams['xtick.labelsize'] = 22 
mpl.rcParams['ytick.labelsize'] = 22 
mpl.rcParams['legend.fontsize'] = 18


np.random.seed(10)

class RESERVOIR():

    def __init__(self, data , n_layers , n_reservoir=200, n_inputs=1 , n_outputs=1 ,
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
        self.n_layers = n_layers
        
    def initweights(self):
        # initialize reservoir weights:
        # begin with a random matrix centered around zero: 
        self.W = np.zeros((self.n_layers , self.n_reservoir , self.n_reservoir))
        for layer in range(self.n_layers-1):
            W = np.random.rand(self.n_reservoir, self.n_reservoir)*2-1
            # delete the fraction of connections given by (self.sparsity):
            for i in range(self.n_reservoir):
                for j in range(self.n_reservoir):
                    if np.random.uniform() < self.sparsity:
                        W[i,j] = 0
                                
            # compute the spectral radius of these weights:
            radius = np.max(np.abs(np.linalg.eigvals(W)))
            # rescale them to reach the requested spectral radius:
            self.W[layer] = W * (self.spectral_radius / radius)

        # random input weights:
        self.W_in = np.zeros((self.n_layers , self.n_reservoir , 1))
        for layer in range(self.n_layers):
            self.W_in[layer]  = np.random.rand(self.n_reservoir , self.n_inputs )*0.2 - 0.1
        # random output weights:
        self.W_out = np.zeros((self.n_layers , self.n_reservoir , 1))
        for layer in range(self.n_layers):
            self.W_out[layer] = np.random.rand(self.n_reservoir , self.n_outputs)*0.2 - 0.1
        # initializing the neurons with ranodm values
        self.state = np.zeros((self.n_layers , self.n_reservoir , 1))
                              
    def _update(self, target_pattern, output_pattern):
        """Updates the output weights"""
        self.W_out += self.learning_rate*((target_pattern - output_pattern)*(self.state) + 0.1*self.W_out)
        
    def _predict(self , input_pattern):
        """Calculates the output pattern"""
        for layer in range(self.n_layers):
            self.state[layer] = np.tanh(np.dot(self.W[layer], self.state[layer])
                                        + np.dot(self.W_in[layer], input_pattern))
            output_pattern = np.dot(np.transpose(self.W_out[layer]) , self.state[layer])[0,0]
            input_pattern = output_pattern
        return output_pattern

    def _train(self):
        """Training the reservoir with 80% of the trajectory"""
        for i in range(int(0.8*len(self.data))):
            output_pattern = self._predict(self.data[i])
            self._update(self.data[i+1], output_pattern)
        
    def _test(self):
        x = self.test_data[0]
        self.prediction = [x]
        for i in range(int(0.2*len(self.data))-1):
            y = self._predict(x)
            self.prediction.append(y)
            x = y
    
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
        