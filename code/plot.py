#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:05:49 2021

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




traj = np.loadtxt('traj1.txt')
x = traj[:,0]
y = traj[:,1]
z = traj[:,2]


print(x[0] , y[0] , z[0])

plt.figure(figsize=(15,10))
plt.plot(x,y)
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('xy.png')

plt.figure(figsize=(15,10))
plt.plot(x,z)
plt.xlabel('X')
plt.ylabel('Z')
plt.savefig('xz.png')

plt.figure(figsize=(15,10))
plt.plot(y,z)
plt.xlabel('Y')
plt.ylabel('Z')
plt.savefig('yz.png')



traj2 = np.loadtxt('traj2.txt')
x2 = traj2[:,0]
y2 = traj2[:,1]
z2 = traj2[:,2]
print(x2[0] , y2[0] , z2[0])


T = np.linspace(0,50 , 2500)

plt.figure(figsize=(20,7))

plt.plot(T , y , label = 'traj1')
plt.plot(T , y2 , label = 'traj2')
plt.legend()
plt.xticks()
plt.ylim(-30,30)



T = np.linspace(0,10 , 500)

plt.figure(figsize=(20,7))

plt.plot(0.906*T , y[2000:] , label = 'traj1')
plt.plot(0.906*T , y2[2000:] , label = 'traj2')
plt.legend()
plt.xticks()
plt.plot(np.ones(15)*0.906 , np.linspace(-30,30 ,15) , 'r--')
plt.ylim(-30,30)
