#a2_5.py
import numpy as np
import sys
import matplotlib.pyplot as plt

def NGP(p,N):

    mesh = np.zeros((N,N,N))
    for i in range(len(p[0])):
        x,y,z = np.round(p[:,i])%N
        mesh[int(x)][int(y)][int(z)] += 1
        
    return mesh

def CiC(p,N):

    mesh = np.zeros((N,N,N))
    for i in range(len(p[0])):
        w = np.zeros(8)
        x,y,z = np.round(p[:,i])%N
        dx,dy,dz = x-p[0,i],y-p[1,i],z-p[2,i]
        sx,sy,sz = np.sign(dx),np.sign(dy),np.sign(dz)
        dx,dy,dz = np.abs(dx),np.abs(dy),np.abs(dz)
        # Calculating all of the weights
        w[0] = (1-dx)*(1-dy)*(1-dz)
        w[1] = (dx)*(1-dy)*(1-dz)
        w[2] = (1-dx)*(dy)*(1-dz)
        w[3] = (1-dx)*(1-dy)*(dz)
        w[4] = (dx)*(dy)*(dz-1)
        w[5] = (dx)*(1-dy)*(dz)
        w[6] = (1-dx)*(dy)*(dz)
        w[7] = (dx)*(dy)*(dz)        
        # Assigning the weights
        mesh[np.int(x)-1][np.int(y)-1][np.int(z)-1] += w[0]
        mesh[np.int(x+sx)-1][np.int(y)-1][np.int(z)-1] += w[1]
        mesh[np.int(x)-1][np.int(y+sy)-1][np.int(z)-1] += w[2]
        mesh[np.int(x)-1][np.int(y)-1][np.int(z+sz)-1] += w[3]
        mesh[np.int(x+sx)-1][np.int(y+sy)-1][np.int(z)-1] += w[4]
        mesh[np.int(x+sx)-1][np.int(y)-1][np.int(z+sz)-1] += w[5]
        mesh[np.int(x)-1][np.int(y+sy)-1][np.int(z+sz)-1] += w[6]   
        mesh[np.int(x+sx)-1][np.int(y+sy)-1][np.int(z+sz)-1] += w[7]

    return mesh



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
