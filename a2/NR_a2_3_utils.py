import numpy as np
import sys
import matplotlib.pyplot as plt

def k_calc(h,f,t,x1,x2,xn):
    # Likely soruce of error: What do we do with the second variable when calculating k?
    # To-do: Try to solve the problem by applying it on a simpler function
    k1 = h * f(t,x1,x2)
    k2 = h * f(t+0.5*h,x1+0.5*k1,x2+0.5*k1)
    k3 = h * f(t+0.5*h,x1+0.5*k2,x2+0.5*k2)
    k4 = h * f(t+h,x1+k3,x2+k3)
    return xn+1/6*k1+1/3*k2+1/3*k3+1/6*k4

def runge_kutta2nd(x1_0,x2_0,t,f1,f2,h=0.01):
    #Implementaiton of the Runge-Kutta method for ODE integration
    # x0,y0 are the starting values and f is the ode()

    t = np.arange(1,t,h)
    x1n,x2n = x1_0,x2_0
    x1_out = np.zeros(len(t))
    for i in range(len(t)):
        x1_new = k_calc(h,f1,t[i],x1n,x2n,x1n)
        x2_new = k_calc(h,f2,t[i],x1n,x2n,x2n)
        x1_out[i] = x1_new
        x1n,x2n = x1_new,x2_new

    return np.sum(x1_out)*h

def runge_kutta(x0,y0,f,xmax,h=0.0001):
    #Implementaiton of the Runge-Kutta method for ODE integration
    # x0,y0 are the starting values and f is the ode()
    xn,yn = x0,y0
    y_out,x_out = [],[]
    while xn < xmax:
        yn_new = k_calc(h,f,xn,yn)
        y_out.append(yn_new)
        xn += h
        x_out.append(xn)
        yn = yn_new

    plt.plot(x_out,y_out)

    return np.sum(y_out)*h
