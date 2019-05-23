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

def random_field_generator(n,N,rng,mu=0):
    # Prepares a random field in Fourier space
    print(f'Generating a random field with n = {n} of dimension {N}x{N} (mu = {mu})')
    df = np.zeros((N,N),dtype=complex)
    # Setting values of top half of the field 
    for j in range((N//2)+1):
    # Determining the value of k_y 
        k_y = j*2*np.pi/N
        for i in range(N):
            # Determining the value of k_x and sigma_x
            if i <= (N//2):
                k_x = (i)*2*np.pi/N
            else:
                k_x = (-N+i)*2*np.pi/N
            # Avoid dividing by 0
            if i != 0 or j != 0:
                sig = ((k_x**2+k_y**2)**0.5)**(n/2)
            else: 
                sig = 0
            # Drawing a random number from normal distrib 
            #df[j][i] = np.random.normal(0,sig)+ 1j*np.random.normal(0,sig)
            rand = box_muller(rng.rand_num(1),rng.rand_num(1),mu,sig)
            df[j][i] = rand[0] + 1j*rand[1]
    # Setting values of points who need to equal their own conjugates
    df[0][0] = 0
    df[0][N//2] = (df[0][N//2].real)**2
    df[N//2][0] = (df[N//2][0].real)**2
    df[N//2][N//2] = (df[N//2][N//2].real)**2
    # Setting values of bottom half of the field using conjugates
    for j in range((N//2)+1):
        for i in range(N):
            df[-j][-i]= df[j][i].conjugate()
    return df
#end random_field generator()

def box_muller(u1,u2,mu,sigma):
    # Implementation of the Box Muller transform
    x1 = (-2*np.log(u1))**0.5*np.sin(2*np.pi*u2)
    x2 = (-2*np.log(u1))**0.5*np.cos(2*np.pi*u2)
    return x1*sigma+mu,x2*sigma+mu
#end box_muller


def random_field_generator_zeld(N,rng,mu=0,sig=1):
    # Prepares a random field in Fourier space
    #print(f'Generating a random field with n = {n} of dimension {N}x{N} (mu = {mu})')
    ck = np.zeros((N,N),dtype=complex)
    Sx = np.zeros((N,N),dtype=complex)
    Sy = np.zeros((N,N),dtype=complex)
    # Setting values of top half of the field 
    for j in range((N//2)+1):
    # Determining the value of k_y 
        k_y = j*2*np.pi/N
        for i in range(N):
            # Determining the value of k_x and sigma_x
            if i <= (N//2):
                k_x = (i)*2*np.pi/N
            else:
                k_x = (-N+i)*2*np.pi/N
            # Drawing a random number from normal distrib
            k = (k_x**2+k_y**2)**0.5 
            if k == 0:
                k = 1
            rand = box_muller(rng.rand_num(1),rng.rand_num(1),mu,sig)
            ck[j][i] = (rand[0]*k**(-3)) - 1j*(rand[1]*k**(-3))
            Sx[j][i] = ck[j][i]*k_x*1j
            Sy[j][i] = ck[j][i]*k_y*1j
    # Setting values of points who need to equal their own conjugates
    ck[0][0] = 0
    Sx[0][0],Sy[0][0] = 0,0
    ck[0][N//2] = (ck[0][N//2].real)**2
    Sx[0][N//2] = (Sx[0][N//2].real)**2
    Sy[0][N//2] = (Sy[0][N//2].real)**2
    ck[N//2][0] = (ck[N//2][0].real)**2
    Sx[N//2][0] = (Sx[N//2][0].real)**2
    Sy[N//2][0] = (Sy[N//2][0].real)**2
    ck[N//2][N//2] = (ck[N//2][N//2].real)**2
    Sx[N//2][N//2] = (Sx[N//2][N//2].real)**2
    Sy[N//2][N//2] = (Sy[N//2][N//2].real)**2
    # Setting values of bottom half of the field using conjugates
    for j in range((N//2)+1):
        for i in range(N):
            ck[-j][-i]= ck[j][i].conjugate()
            Sx[-j][-i]= Sx[j][i].conjugate()
            Sy[-j][-i]= Sy[j][i].conjugate()
    return ck,Sx,Sy
#end random_field generator()
