import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable

from torchvision import transforms, datasets



# Helper functions
def seqpairs(input_list):
    for i in range(len(input_list) - (2 - 1)):
        yield input_list[i:i+2]

def moving_average(a, n=100):
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Hinge loss (why doesn't pytorch have this...)
class HingeLoss(torch.nn.Module):

    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, output, target):

        hinge_loss = 1 - torch.mul(output, target)
        hinge_loss[hinge_loss < 0] = 0
        return hinge_loss



def run_rnn(Ntrials = 500, Nreps = 1, dt = .005, lr = .05, blob_scale = 1., gamma=0.01, alpha=.1, data_type = 'gauss', exptname="expt"):

    dtype = torch.FloatTensor

    # Experiment parameters
    D = 6.370
    Dp = 3.136-D
    T0 = .16

    # Simulation timestep
   # dt = .005 # .0005
    print(data_type)

    if data_type == 'gauss':

        # Network parameters
        layer_sizes = [1, 1]
        input_size = layer_sizes[0]
        output_size = layer_sizes[-1]

        #lr=.05
        #gamma = .01

        #DT_target = .9-T0
        lr=.1
        gamma = .01
        alpha=.05

        z0 = 30.
        w0 = .0001

        # Dataset parameters
        A = .82
        ci = .01 
        co = 32

        sample_input = lambda ys: torch.Tensor(np.random.normal(ys*A*dt/input_size,ci*np.sqrt(dt/input_size),(input_size,))).type(dtype)


    elif data_type == 'gauss_multi':

        # Network parameters
        layer_sizes = [16, 1]
        input_size = layer_sizes[0]
        output_size = layer_sizes[-1]

        #lr=.5
        #gamma = .01

        #DT_target = .9-T0

        z0 = 30.
        w0 = .0001

        # Dataset parameters
        A = 1.
        ci = .5 
        co = 30

        sample_input = lambda ys: torch.Tensor(np.random.normal(ys*A*dt/input_size,ci*np.sqrt(dt/input_size),(input_size,))).type(dtype)

    elif data_type == 'blob':

    
        # Network parameters
        layer_sizes = [20*35, 1]
        input_size = layer_sizes[0]
        output_size = layer_sizes[-1]

        lr=lr/(20*45)#.5/(20*35)
        #gamma = .01

        #DT_target = .9-T0

        z0 = 30.
        w0 = .0001

        # Dataset parameters
        co = 30

        # For theory only
        A = 1.
        ci = .5 

        data_transform = transforms.Compose([
            transforms.Resize(20),
            transforms.Grayscale(),
            transforms.RandomAffine(20, translate=(.25,.25), scale=None),
            transforms.ToTensor()
        ])

        blob_dataset = datasets.ImageFolder(root='blobs',  transform=data_transform)

        sample_input = lambda ys: blob_dataset[ys>0][0].view(20*35)*dt*blob_scale

    else:
        print(data_type)
        raise ValueError('Dataset type not recognized')



    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(w0)

    # History variables to store outcomes
    print( "Run simulations")
    w_history = np.zeros((Ntrials, Nreps))
    e_history = np.zeros((Ntrials, Nreps))
    ER_a = np.zeros((Ntrials, Nreps))
    DT = np.zeros((Ntrials, Nreps))
    TT = np.zeros((Ntrials, Nreps))

    # Threshold
    z = np.zeros((Ntrials+1, Nreps))
    z[0,:] = z0

    for r in range(Nreps):
        print( "Net number", r+1, " of ", Nreps)

        # Create network
        modules = []
        for in_sz, out_sz in seqpairs(layer_sizes):
            modules.append(nn.Linear(in_sz, out_sz, bias=False))
        net = nn.Sequential(*modules)

        # Initialize weights
        net.apply(init_weights)

        loss = HingeLoss() #nn.SoftMarginLoss()#nn.BCEWithLogitsLoss() #nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr)

        for trial in range(Ntrials):

            if trial%1000==0:
                print( trial)

            # Trial type
            y = (np.random.rand()>.5)#*2-1 
            ys = 2*y-1
            yt = torch.Tensor([ys])

            # Roll out in time
            optimizer.zero_grad()
            t = 0
            total_out = 0
            while abs(total_out) < z[trial,r]:

                x = sample_input(ys)
                out = net(x)

                n = torch.Tensor(np.random.normal(0,co*np.sqrt(dt),(1,))).type(dtype)

                total_out = total_out + out + n
                t = t + 1


            total_loss = loss(total_out, yt) 

            # Gradient descent
            total_loss.backward()
            optimizer.step()

            # Store trial results
            e_history[trial,r] = np.sign(total_out.detach()) != ys
            
            if trial > 0:
                ER_a[trial,r] = (1-alpha)*ER_a[trial-1,r]+alpha*e_history[trial,r]
            else:
                ER_a[trial,r] = .45
                
            DT[trial,r] = (t-1)*dt
            TT[trial,r] = DT[trial,r] + T0 + D + Dp*e_history[trial,r]
            w_history[trial,r] = np.sum(np.abs(net[0].weight.data.numpy()))#net[0].weight.data[0][0]

            # Set DT target
            DT_target = .9-T0#(1/(1/( ER_a[trial,r]*np.log( (1-ER_a[trial,r])/ER_a[trial,r] ) ) + 1/ (1 - 2*ER_a[trial,r]) ))*(D + Dp + T0)
   
            # Update threshold to track desired decision time
            z[trial+1,r] = z[trial,r] + gamma*(DT_target - DT[trial,r])


    
    time = np.cumsum(TT,axis=0)
    np.savez(exptname, time=time, e_history=e_history, ER_a=ER_a, DT=DT, TT=TT, w_history=w_history, z=z, co=co, dt=dt, Ntrials=Ntrials, lr=lr, gamma=gamma, alpha=alpha, DT_target=DT_target, z0=z0, w0=w0)
    
