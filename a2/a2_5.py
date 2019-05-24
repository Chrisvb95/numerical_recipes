#a2_5.py
mport numpy as np
import sys
import matplotlib.pyplot as plt

def fft1D(x,Nj,start=0,step=1):
    #if Nj%2 > 0:
    #    raise ValueError('Size of input array must be a power of 2')
    if Nj == 1: 
        return [x[start]]
    new_step = step*2
    hNj = Nj//2
    rs = fft1D(x,hNj,start,new_step)+fft1D(x,hNj,start+step,new_step)
    for i in range(hNj):
        rs_new[i],rs_new[i+hNj]=rs[i]+np.exp(-2j*np.pi*i/Nj)*rs[i+hNj],rs[i]-np.exp(-2j*np.pi*i/Nj)*rs[i+hNj]
        res_new[i] = rs[i]+np.exp(-2j*np.pi*i/Nj)*rs[i+hNj]
        res_new[i+hNj] = rs[i]-np.exp(-2j*np.pi*i/Nj)*rs[i+hNj]
    return rs

def fft2D(x):
    if Nj%2 > 0:
        raise ValueError('Size of input array must be a power of 2')
    x = np.array(x,dtype=complex)
    if len(x.shape) == 2:
        for i in range(x.shape[1]):
            x[:,i] = fft1D(x[:,i],len(x[1]))
        for j in range(x.shape[0]):
            x[j] = fft1D(x[j],len(x[0]))
        return x

def fft3D(x):
    if Nj%2 > 0:
        raise ValueError('Size of input array must be a power of 2')
    x = np.array(x,dtype=complex)
    for i in range(x.shape[1]):
        x[:,i] = fft2D(x[:,i],len(x[1]))
    for j in range(x.shape[0]):
        x[j] = fft2D(x[j],len(x[0]))
    return x
