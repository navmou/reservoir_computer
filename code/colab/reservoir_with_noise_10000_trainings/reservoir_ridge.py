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




class RESERVOIR():

    def __init__(self, data , n_reservoir=200, n_inputs=1 , n_outputs=1 ,
                 spectral_radius = 0.95, sparsity=0 , learning_rate=0.2 , n_trainings = 100):
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
        self.n_trainings = n_trainings
        
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
        for i in range(self.n_reservoir):
            if np.random.uniform() < 0:
                self.W_in[i] = 0
        self.W_in = self.W_in/(np.linalg.norm(self.W_in)+0.001)
        # random output weights:
        self.W_out = np.random.rand(self.n_reservoir , self.n_outputs)*0.2 - 0.1
        self.W_out = self.W_out/(np.linalg.norm(self.W_out)+0.001)
        # initializing the neurons with ranodm values
        self.state = np.zeros((self.n_reservoir , 1))
        self.R = self.state
                              
    def _update(self, target_pattern, output_pattern):
        """Updates the output weights"""
        self.W_out += self.learning_rate*(target_pattern - output_pattern)*(self.state)
        
    def _update_state(self , input_pattern):
        """Calculates the output pattern"""
        self.state = (np.tanh(np.dot(self.W, self.state) + np.dot(self.W_in, input_pattern)) 
        + 0.001*(np.random.rand(self.n_reservoir,1)-0.5))
        return np.dot(np.transpose(self.state) , self.W_out)[0,0]
        #return self.clf.predict(self.state.reshape(1,self.n_reservoir))


    def _train(self):
        #self.clf = Ridge(alpha=0.1)
        for train in range(self.n_trainings):
            #print(np.linalg.norm(self.W_out))
            #self.R = np.zeros((2000,self.n_reservoir))
            traj = [self.data[0]]
            for i in range(2000):
                #self.R[i] = self.state.reshape(1,self.n_reservoir)
                traj.append(self._update_state(self.data[i]))
                self._update(self.data[i+1], traj[-1])
                #self.W_out = self.W_out/np.linalg.norm(self.W_out)
                #self.clf.fit(self.state.reshape(1,self.n_reservoir) , np.array([self.data[i+1]]))
                #self.W_out = self.clf.coef_.reshape(self.n_reservoir,1)
                
            #print(self.R)
            #print(self.R.shape)
            #self.clf.fit(self.R, self.data[1:2001])
            #self.W_out = (self.clf.coef_.reshape(self.n_reservoir , 1))
            if train%100==0:
                print(f"train: {train} , learning_rate: {self.learning_rate}")
                self._test()
                self._plot(False)
                if train > 5000:
                    self.learning_rate = 0.001
                if train > 7000:
                    self.learning_rate = 0.0005
            #    print(f'training {train} is doen!')
            #    plt.figure(figsize=(20,7))
            #    plt.plot(traj, label='train')
            #    plt.plot(self.data[:2000] , label='goal')
            #    plt.legend()
            #    plt.show()
                #self.learning_rate = np.max([self.learning_rate*0.9  , 0.0001])
                #print(np.sum(np.abs(self.clf.coef_.reshape(self.n_reservoir , 1) - self.W_out))) 
        
    def _predict(self, input_pattern):
        self.state = np.tanh(np.dot(self.W, self.state) + np.dot(self.W_in, input_pattern))
        return np.dot(np.transpose(self.W_out),self.state)
        #return self.clf.predict(self.state.T)[0]

    def _test(self):
        x = self.test_data[0]
        self.prediction = [x]
        for i in range(int(0.2*len(self.data))-1):
            self.prediction.append(self._predict(self.prediction[-1]))
    
    def _plot(self, saving):
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
        if saving:
            plt.savefig('comparison.png')
        else:
            plt.show()
        
    
    def _save(self):
        with open('W_out.npy' , 'wb') as f:
            np.save(f, self.W_out)
        with open('W_in.npy' , 'wb') as f:
            np.save(f, self.W_in)
        with open('W.npy' , 'wb') as f:
            np.save(f, self.W)
        with open('state.npy' , 'wb') as f:
            np.save(f, self.state)
        

    def run(self):
        #for repeat in range(20,100):
        np.random.seed(66)
        self.initweights()
        self._train()
        self._test()
        #print(f'repeat : {repeat}')
        self._plot(True)
        self._save()
    
    