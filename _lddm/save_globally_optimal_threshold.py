# This script calculates an approximately globally optimal threshold trajectory using gradient descent

# Load pytorch 

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

execfile("reduction.py")

dt = 160
Nsteps = 5000
time = np.arange(Nsteps)*dt

A = 0.9542#/np.random.uniform(0,1)
ci = 0.3216#/np.random.uniform(0,4)
co = 30#*np.random.uniform(0,2)
w0 = .0001

lr = 4.5942/100.

z0=30

Nitr = 1000

optim_lr = 50

ER_vec = np.linspace(0.0001,.5-.0001,100)
Dnorm_opc = 1/(1/( ER_vec*np.log( (1-ER_vec)/ER_vec ) ) + 1/ (1 - 2*ER_vec) )

z   = [Variable(z0*torch.ones(1), requires_grad = True) for t in np.arange(Nsteps)]
w   = [Variable(w0*torch.ones(1), requires_grad = False) for t in np.arange(Nsteps)]

optimizer = optim.SGD(z, lr=optim_lr, momentum=.9)

time = np.arange(Nsteps)*dt

Rtot_hist = np.zeros(Nitr)

for b in range(Nitr):
    print b
    optimizer.zero_grad()

    Rtot = 0
    for t in range(Nsteps):
        dW, RR = dyn_update(A, ci, co, lr, z[t], w[t])
        if t < Nsteps-1:
            w[t+1] = w[t] + dW*dt 
            Rtot = Rtot + RR*dt


    Rtot_hist[b] = Rtot.data.numpy()
    Rtot = -Rtot
    Rtot.backward()
    optimizer.step()

np.savez('globally_optimal_thres.npz',z=numpify(z))

