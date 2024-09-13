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
mpl.rcParams['legend.fontsize'] = 22




class RESERVOIR():

    def __init__(self, data , n_reservoir=200, n_inputs=1 , n_outputs=1 ,
                 spectral_radius = 0.95, sparsity=0 , learning_rate=0.2 , n_trainings = 100):
        """
        Args:
            data: input time series consisting training and testing sets
            n_reservoir: number of reservoir neurons
            spectral_radius: spectral radius of the recurrent weight matrix
            sparsity: proportion of recurrent weights set to zero
            learning_rate: the learning rate to train using back-propagation
            n_trainings: number of trainings for the back propagion training
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
        self.J = 1
        self.landas = []
                              
    def _update_weight(self, target_pattern, output_pattern):
        """Updates the output weights with gradient descent"""
        self.W_out += self.learning_rate*(target_pattern - output_pattern)*(self.state)
    
    def _predict(self, input_pattern):
        """given an input predicts the outpu using the trained weight"""
        b = np.dot(self.W, self.state) + np.dot(self.W_in, input_pattern)
        self.state = np.tanh(b)
        self.D = np.diag(1-np.tanh(b[:,0])**2)
        return np.dot(np.transpose(self.W_out),self.state)

    def _train(self, ridge=True):
        if ridge:
            self.model = Ridge(alpha=0.1)
            self.R = np.zeros((2000,self.n_reservoir))
            self.traj = [self.data[0]]
            for i in range(2000):
                self.R[i] = self.state.reshape(1,self.n_reservoir)
                self.traj.append(self._predict(self.data[i]))                
                #self._single_val()
                #if i %100 == 0:
                #    print(i)

            self.model.fit(self.R, self.data[:2000])
            self.W_out = (self.model.coef_.reshape(self.n_reservoir , 1))
            self._test()
            self._plot(True)

        else:
            for train in range(self.n_trainings):
                self.R = np.zeros((2000,self.n_reservoir))
                self.traj = [self.data[0]]
                for i in range(2000):
                    self.R[i] = self.state.reshape(1,self.n_reservoir)
                    self.traj.append(self._predict(self.data[i]))
                    self._update_weight(self.data[i+1], self.traj[-1])
                if train%1==0:
                    print(f"train: {train} , learning_rate: {self.learning_rate}")
                    self._test()
                    self._plot(False)
                    self._plot_train(train)


    def _test(self):
        x = self.test_data[0]
        self.prediction = [x]
        for i in range(int(0.2*len(self.data))-1):
            self.prediction.append(self._predict(self.prediction[-1]))
    
    def _single_val(self):
        self.J = np.dot(np.dot(self.D , self.W) , self.J)
        self.landas.append(np.linalg.svd(self.J)[1][0])


    def _plot(self, saving):
        landa1 = 0.906
        T = np.linspace(0,10,500)*landa1
        plt.figure(figsize=(25,7))
        plt.plot(T , self.prediction , label='Prediction')
        plt.plot(T , self.test_data , label='Actual')
        plt.plot(np.ones(15)*landa1 , np.linspace(-30,30,15) , 'r--' , label='Lyaponov time')
        plt.legend(loc=4)
        plt.ylim(-30,30)
        plt.xlabel(r'$\lambda_1 t$')
        plt.ylabel(r'$y(t)$')
        if saving:
            plt.savefig('comparison.png')
        else:
            plt.show()
        
    def _plot_train(self, train):
        print(f'training {train} is doen!')
        plt.figure(figsize=(20,9))
        plt.plot(self.traj, label='train')
        plt.plot(self.data[:2000] , label='goal')
        plt.legend()
        plt.show()

    def _plot_lyapunov(self):
        self.landas = np.array(self.landas)
        plt.figure(figsize=(15,7))
        n = np.arange(1,len(self.landas)+1)
        plt.plot(n , 1/n*np.log(self.landas))
        plt.xlabel('n')
        plt.ylabel(r'$\frac{1}{n}\log \Lambda_1 (n)$')
        plt.savefig('Lyapunov.png')


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
        for repeat in range(66,67):
            print(f'random seed : {repeat}')
            np.random.seed(repeat)
            self.initweights()
            self._train()
            #self._test()
            #print(f'repeat : {repeat}')
            #self._plot(True)
            #self._save()
            #self._plot_lyapunov()

    
