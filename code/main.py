#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 19:08:58 2021

@author: navid
"""


import numpy as np

from reservoir import RESERVOIR

data = np.loadtxt('traj1.txt')
x = data[:,0]
y = data[:,1]
z = data[:,2]

res = RESERVOIR(data = y , n_reservoir=1000 , n_inputs=1 , n_outputs=1 , 
                spectral_radius=1.2 , sparsity=0.1 , learning_rate=0.002)# , beta = 0 , n_trainings=1000)
res.run()