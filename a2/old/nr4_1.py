import numpy as np 
import matplotlib.pyplot as plt


N = 1e6

def xsquared_sinx(x):
    return x**2*np.sin(x)

#1.a 
#Calculate the analytical derivative of this function and plot the function in the range 0,2pi

x = np.linspace(0,2*np.pi,10000)
plt.plot(x,2*x*np.sin(x)+(x**2)*np.cos(x))
plt.show()

#1.b 
#
