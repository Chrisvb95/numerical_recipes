#NR_a1_main.py

import numpy as np 
import matplotlib.pyplot as plt 
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
dndx = utils.ridders_diff(n,b)
print(dndx)
x = b 
dndx_analitic = (A*100) * ((a-3)*(x/b)**(a-4)*np.exp(-(x/b)**c)) + ((x/b)**(a-3)*-(c/x)*(x/b)**c*np.exp(-(x/b)**c))
print(dndx_analitic)
dndx_analitic = (A*100) * ((a-3)*(x/b)**(a-4)*np.exp(-(x/b)**c))/b - ((c*np.exp(-(x/b)**c)*(x/b)**(a+c-4))/b) 
print(dndx_analitic)

y = lambda x: np.sin(x)
dydx = utils.ridders_diff(y,np.array([np.pi/4]))
print(dydx)