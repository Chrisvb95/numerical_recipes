#NR_a1_main.py
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import sys
import NR_a1_utils as utils 
from importlib import reload
reload(utils)

seed = 42
print('Original seed:',seed)
'''
# Poisson distribution
a1 = [[1,0],[5,10],[3,20],[2.6,40]]#,[101,200]]
for i in range(len(a1)):
	print(utils.poisson_distribution(a1[i][0],a1[i][1]))
a1 = [[1,0],[5,10],[3,20],[2.6,40],[101,200]]
#a1 = [[101,200]]
print('I hate my life')
for i in range(len(a1)):
	print(utils.poisson_distribution_new(a1[i][0],a1[i][1]))
'''
#--- 1.b ---
# RNG 
rng = utils.rng(seed)
# Scatter plot
n = 1000 
rand = rng.rand_num(n)
#plt.scatter(rand[:(len(rand)-1)],rand[1:])
#plt.title('Sequential number plot for {} random numbers with seed {}'.format(1000,seed))
#plt.show()
# Histogram
n = 1000000
rand = rng.rand_num(n)
#plt.hist(rand,bins=20,range=(0,1))
#plt.show()

#--- 2.a --- 
a = rng.rand_num(1,min=1.1,max=2.5)
b = rng.rand_num(1,min=0.5,max=2)
c = rng.rand_num(1,min=1.5,max=4)

f = lambda x: 4*np.pi* (x**(a-1))/(b**(a-3)) *np.exp(-(x/b)**c)
f_int = utils.romber_int(f,0,5)
A = 1/f_int 
print('A = {}; a,b,c = {},{},{}'.format(A,a,b,c))

#--- 2.c --- 
n = lambda x: A*100*(x/b)**(a-3)*np.exp(-(x/b)**c)
#plt.plot(x,pn(x))
#plt.plot(x,g)
#plt.yscale('log')
#plt.xlim(0,5)
#plt.show()

dndx = utils.ridders_diff(n,b)
print('{0:.12f}'.format(float(dndx)))
x = b 
dndx_analitic = (A*100) * ((a-3)*(x/b)**(a-4)*np.exp(-(x/b)**c)) + ((x/b)**(a-3)*-(c/x)*(x/b)**c*np.exp(-(x/b)**c))
print('{0:.12f}'.format(float(dndx_analitic)))
dndx_analitic = (A*100) * ((a-3)*(x/b)**(a-4)*np.exp(-(x/b)**c))/b - ((c*np.exp(-(x/b)**c)*(x/b)**(a+c-4))/b) 
print('{0:.12f}'.format(float(dndx_analitic)))

y = lambda x: np.sin(x)
dydx = utils.ridders_diff(y,np.array([np.pi/4]))
print('{0:.12f}'.format(float(dydx)))

#--- 2.d --- 
N = 100
# Drawing random samples from n(x)
pn = lambda x: (n(x)*4*np.pi*x**2)/100
x_p = np.linspace(0,5,200)
g = np.max(pn(x_p)[1:])+0.01
samples = utils.rejection_sampler(N,pn,5,g,rng)
r = samples[0]
# Generating random angles: 
phi = rng.rand_num(N,min=0,max=2*np.pi)
theta = np.arccos(2*rng.rand_num(N)-1)
x,y,z = r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)
# Plotting positions for N galaxies
ax = plt.figure().add_subplot(111,projection='3d')
ax.scatter(x,y,z)
plt.show()

#--- 2.e --- 