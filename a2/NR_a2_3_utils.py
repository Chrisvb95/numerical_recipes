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

def k_calc2nd(h,f,g,t,x1,x2):
    # Support function for runge_kutta method (for 2nd order ODEs)
    k1 = h * f(t,x1,x2)
    l1 = h * g(t,x1,x2)
    k2 = h * f(t+0.5*h,x1+0.5*k1,x2+0.5*l1)
    l2 = h * g(t+0.5*h,x1+0.5*k1,x2+0.5*l1)
    k3 = h * f(t+0.5*h,x1+0.5*k2,x2+0.5*k2)
    l3 = h * g(t+0.5*h,x1+0.5*k2,x2+0.5*k2)
    k4 = h * f(t+h,x1+k3,x2+k3)
    l4 = h * g(t+h,x1+k3,x2+k3)

    x1_new = x1+1/6*(k1+2*k2+2*k3+k4)
    x2_new = x2+1/6*(l1+2*l2+2*l3+l4)

    return x1_new, x2_new
#end k_calc2nd()

def runge_kutta2nd(x1_0,x2_0,t0,t,f,g,h=0.01):
    #Implementaiton of the Runge-Kutta method for ODE integration
    # x0,y0 are the starting values and f is the ode()
    t = np.arange(t0,t+h,h)
    #print(t)
    x1n,x2n = x1_0,x2_0
    x1_out = np.zeros(len(t))
    for i in range(len(t)):
        x1n,x2n = k_calc2nd(h,f,g,t[i],x1n,x2n)
        x1_out[i] = x1n 
    return np.sum(x1_out)*h,x1_out
