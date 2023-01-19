import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Experimental parameters
D = 6.370
Dp = 3.136-D
T0 = .16

def RR_calc(A, ci, co, z, w):
    zbar = z/(A*w)
    SNR =  A**2*w**2/(ci**2*w**2 + co**2)
    ER = 1/(1 + torch.exp(2*zbar*SNR))
    DT = zbar*torch.tanh(zbar*SNR)
    TT = DT + D + T0 + Dp*ER
    RR = (1-ER)/TT
    
    return RR

def dyn_update(A, ci, co, lr, z, w):
    zbar = z/(A*w)
    SNR =  A**2*w**2/(ci**2*w**2 + co**2)
    ER = 1/(1 + torch.exp(2*zbar*SNR))
    DT = zbar*torch.tanh(zbar*SNR)
    TT = DT + D + T0 + Dp*ER
    RR = (1-ER)/TT
    dW = lr*ER*( (A*DT) + 1./(1+co**2/(w**2*ci**2))*(-z/w - A*DT))/TTs

    return dW, RR

def const_zpol(z,w,zbar,SNR,ER,DT,TT,RR,t,dt,z0):
    return z0

def descriptive_zpol(z,w,zbar,SNR,ER,DT,TT,RR,t,dt,A,gamma):
    DT_target = (1/(1/( ER*np.log( (1-ER)/ER ) ) + 1/ (1 - 2*ER) ))*(D + Dp + T0)
    return  torch.max(torch.ones(1),z + gamma*(-z + DT_target/(1/(A*w)*torch.tanh(zbar*SNR)))/TT*dt)

def compute_timecourse_rl(A, ci, co, w0, z0, lr, dt, Nsteps):
    zbar= [Variable(torch.ones(1), requires_grad = False) for t in np.arange(Nsteps)]
    SNR = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    ER  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    DT  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    TT  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    RR  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    w   = [Variable(w0*torch.ones(1), requires_grad = False) for t in np.arange(Nsteps)]
    z   = [Variable(z0*torch.ones(1), requires_grad = False) for t in np.arange(Nsteps)]
    Rtot= [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    
    time = np.arange(Nsteps)*dt
    
    for t in range(Nsteps):
        zbar[t] = z[t]/(A*w[t])
        SNR[t] =  A**2*w[t]**2/(ci**2*w[t]**2 + co**2)
        ER[t] = 1/(1 + torch.exp(2*zbar[t]*SNR[t]))
        DT[t] = zbar[t]*torch.tanh(zbar[t]*SNR[t])
        TT[t] = DT[t] + D + T0 + Dp*ER[t]
        RR[t] = (1-ER[t])/TT[t]
        
        
        if t < Nsteps-1:
            
            first_term = (1-torch.exp(-2*zbar[t]*SNR[t]))
            second_term = 2*(D+T0+Dp - zbar[t])*torch.exp(-2*zbar[t]*SNR[t])
            
            w[t+1] = w[t] + lr*RR[t]**2*( first_term*z[t]/(A*w[t]**2) - second_term*(z[t]*A/ci**2)/(w[t] + co**2/(ci**2*w[t]))**2*(1 - co**2/(ci**2*w[t]**2)) )
            Rtot[t+1] =  Rtot[t] + RR[t]*dt
            
            z[t+1] = z[t] - lr*RR[t]**2*( first_term/(A*w[t]) - second_term*(A/ci**2)/(w[t] + co**2/(ci**2*w[t])) )  
            
            #z[t+1] = z[t] + lr*( -RR[t]**2*( 1./(A*w[t])*(1.-torch.exp(-2.*zbar[t]*SNR[t]))) - 2.*(D + T0+Dp-zbar[t])*torch.exp(-2*zbar[t]*SNR[t])*(A/ci**2)/(w[t]+co**2/(ci**2*w[t])) ) 
            
            #RR_component = 1/(A*w[t])*(1-2*ER[t])/(1-ER[t])*(A/ci**2)/(w[t]+co**2/(ci**2*w[t]))
            
            #RR_component = 1/(A*w[t])*(1-torch.exp(-2*zbar[t]*SNR[t])) - 2*(D+T0+Dp - zbar[t])*torch.exp(-2*zbar[t]*SNR[t])*(A/ci**2)/(w[t]+co**2/(ci**2*w[t]))
                                              
            #RR_gradient_wrt_z = -RR[t]**2*RR_component
            #z[t+1] = z[t] + lr*RR_gradient_wrt_z
            
            #RR_component = 1./(A.*u).*(1-2.*ER)./(1-ER) ...
            #            - 2*(D+T0+Dp - z./(A.*u)).*ER./(1-ER).*(Abs/A)./(u+c./u);
                       
    
    return ER, DT, RR, w, z, Rtot


def compute_timecourse_zpolicy(A, ci, co, w0, z0, zpol, lr, dt, Nsteps):
    zbar= [Variable(torch.ones(1), requires_grad = False) for t in np.arange(Nsteps)]
    SNR = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    ER  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    DT  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    TT  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    RR  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    w   = [Variable(w0*torch.ones(1), requires_grad = False) for t in np.arange(Nsteps)]
    Rtot= [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    z = [Variable(z0*torch.ones(1), requires_grad = False) for t in np.arange(Nsteps)]
        
    time = np.arange(Nsteps)*dt
    
    for t in range(Nsteps):
        zbar[t] = z[t]/(A*w[t])
        SNR[t] =  A**2*w[t]**2/(ci**2*w[t]**2 + co**2)
        ER[t] = 1/(1 + torch.exp(2*zbar[t]*SNR[t]))
        DT[t] = zbar[t]*torch.tanh(zbar[t]*SNR[t])
        TT[t] = DT[t] + D + T0 + Dp*ER[t]
        RR[t] = (1-ER[t])/TT[t]
        
        
        if t < Nsteps-1:
            w[t+1] = w[t] + lr*ER[t]*( (A*DT[t]) + 1./(1+co**2/(w[t]**2*ci**2))*(-z[t]/w[t] - A*DT[t]))/TT[t]*dt 
            Rtot[t+1] =  Rtot[t] + RR[t]*dt
            z[t+1] = zpol(z[t],w[t],zbar[t],SNR[t],ER[t],DT[t],TT[t],RR[t],t,dt)
            
    
    return ER, DT, RR, w, Rtot
    
def compute_timecourse_ztraj(A, ci, co, w0, z, lr, dt, Nsteps):
    zbar= [Variable(torch.ones(1), requires_grad = False) for t in np.arange(Nsteps)]
    SNR = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    ER  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    DT  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    TT  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    RR  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    w   = [Variable(w0*torch.ones(1), requires_grad = False) for t in np.arange(Nsteps)]
    Rtot= [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    
    time = np.arange(Nsteps)*dt
    
    for t in range(Nsteps):
        zbar[t] = z[t]/(A*w[t])
        SNR[t] =  A**2*w[t]**2/(ci**2*w[t]**2 + co**2)
        ER[t] = 1/(1 + torch.exp(2*zbar[t]*SNR[t]))
        DT[t] = zbar[t]*torch.tanh(zbar[t]*SNR[t])
        TT[t] = DT[t] + D + T0 + Dp*ER[t]
        RR[t] = (1-ER[t])/TT[t]
        
        
        if t < Nsteps-1:
            w[t+1] = w[t] + lr*ER[t]*( (A*DT[t]) + 1./(1+co**2/(w[t]**2*ci**2))*(-z[t]/w[t] - A*DT[t]))/TT[t]*dt 
            Rtot[t+1] =  Rtot[t] + RR[t]*dt
            
    
    return ER, DT, RR, w, Rtot

def compute_timecourse(A, ci, co, w0, z0, lr, gamma, DT_target, dt, Nsteps):
    
    zbar= [Variable(torch.ones(1), requires_grad = False) for t in np.arange(Nsteps)]
    SNR = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    ER  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    DT  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    TT  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    RR  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    w   = [Variable(w0*torch.ones(1), requires_grad = False) for t in np.arange(Nsteps)]
    z   = [Variable(z0*torch.ones(1), requires_grad = False) for t in np.arange(Nsteps)]
    trad = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
 
    Rtot = 0
    for t in range(Nsteps):
        zbar[t] = z[t]/(A*w[t])
        SNR[t] =  A**2*w[t]**2/(ci**2*w[t]**2 + co**2)
        ER[t] = 1/(1 + torch.exp(2*zbar[t]*SNR[t]))
        DT[t] = zbar[t]*torch.tanh(zbar[t]*SNR[t])
        TT[t] = DT[t] + D + T0 + Dp*ER[t]
        RR[t] = (1-ER[t])/TT[t]
        Rtot =  Rtot + RR[t]*dt
        trad[t] = 1./(1+co**2/(w[t]**2*ci**2))
        DT_target = (1/(1/( ER[t]*np.log( (1-ER[t])/ER[t] ) ) + 1/ (1 - 2*ER[t]) ))*(D + Dp + T0)
        if t < Nsteps-1:
            
            w[t+1] = w[t] + lr*ER[t]*( (A*DT[t]) + 1./(1+co**2/(w[t]**2*ci**2))*(-z[t]/w[t] - A*DT[t]))/TT[t]*dt 
            # Linearization via nonlinear feedback: dynamics linear in DT, nonlinear in z
            z[t+1] = z[t] + gamma*(-z[t] + DT_target/(1/(A*w[t])*torch.tanh(zbar[t]*SNR[t])))/TT[t]*dt
            #z[t+1] = z[t] + gamma*(DT_target - DT[t])/TT[t]*dt
    time = np.arange(Nsteps)*dt
    return ER, DT, RR, w, Rtot, trad
   
def compute_timecourse_opc(A, ci, co, w0, z0, lr, gamma, dt, Nsteps):
    
    zbar= [Variable(torch.ones(1), requires_grad = False) for t in np.arange(Nsteps)]
    SNR = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    ER  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    DT  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    TT  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    RR  = [Variable(torch.zeros(1), requires_grad = False) for t in np.arange(Nsteps)]
    w   = [Variable(w0*torch.ones(1), requires_grad = False) for t in np.arange(Nsteps)]
    z   = [Variable(z0*torch.ones(1), requires_grad = False) for t in np.arange(Nsteps)]

    Rtot = 0
    for t in range(Nsteps):
        zbar[t] = z[t]/(A*w[t])
        SNR[t] =  A**2*w[t]**2/(ci**2*w[t]**2 + co**2)
        ER[t] = 1/(1 + torch.exp(2*zbar[t]*SNR[t]))
        DT[t] = zbar[t]*torch.tanh(zbar[t]*SNR[t])
        TT[t] = DT[t] + D + T0 + Dp*ER[t]
        RR[t] = (1-ER[t])/TT[t]
        Rtot =  Rtot + RR[t]*dt
        DT_target = 1/(1/( ER[t]*np.log( (1-ER[t])/ER[t] ) ) + 1/ (1 - 2*ER[t]) )
        if t < Nsteps-1:
            w[t+1] = w[t] + lr/100.*ER[t]*( (A*DT[t]) + 1./(1+co**2/(w[t]**2*ci**2))*(-z[t]/w[t] - A*DT[t]))/TT[t]*dt
            z[t+1] = z[t] + gamma/100.*(DT_target - DT[t])/TT[t]*dt
    time = np.arange(Nsteps)*dt
    return ER, DT, RR, z, DT_target
    
    
def numpify(x): 
    return np.vstack([t.data.numpy() for t in x])