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
N = 1000 
rand = rng.rand_num(N)
#plt.scatter(rand[:(len(rand)-1)],rand[1:])
#plt.title('Sequential number plot for {} random numbers with seed {}'.format(1000,seed))
#plt.show()
# Histogram
N = 1000000
rand = rng.rand_num(N)
#plt.hist(rand,bins=20,range=(0,1))
#plt.show()

#--- 2.a --- 
a = rng.rand_num(1,min=1.1,max=2.5)
b = rng.rand_num(1,min=0.5,max=2)
c = rng.rand_num(1,min=1.5,max=4)
f = lambda x: 4*np.pi* (x**(a-1))/(b**(a-3)) *np.exp(-(x/b)**c)
f_int = utils.romber_int(f,0,5)
A = 1/f_int 
print('A = {}; a,b,c = {},{},{}'.format(A,float(a),float(b),float(c)))
A = 0.03

#--- 2.c --- 
n = lambda x: A*100*(x/b)**(a-3)*np.exp(-(x/b)**c)
x = b
dndx = utils.ridders_diff(n,np.array([b]))
dndx_analitic = lambda x: (A*100) * (((a-3)*(x/b)**(a-4)*np.exp(-(x/b)**c))/b - ((c*np.exp(-(x/b)**c)*(x/b)**(a+c-4))/b)) 
dndx_an = dndx_analitic(x)
print('dn/dx at x = b: analytic = {0:.12f}; numerical = {1:.12f}'.format(float(dndx_an),float(dndx)))

#--- 2.d --- 
N = 100
xmax = 5
# Drawing random samples from n(x)
pn = lambda x: (n(x)*4*np.pi*x**2)/100
x_p = np.linspace(0,xmax,200)
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
'''
N = 100000
samples = utils.rejection_sampler(N,pn,5,g,rng)
r = samples[0]
bins = np.logspace(np.log10(1e-4),np.log10(xmax),num=20)
plt.hist(r,bins=bins,range=(1e-4,xmax),density=True)
plt.plot(bins,pn(bins))
plt.yscale('log')
plt.xscale('log')
plt.show()
'''
#--- 2.f --- 
x = np.linspace(0,xmax*2,500)
plt.plot(x,2*pn(x))
#plt.yscale('log')
#plt.xscale('log')
plt.show()

dpndx = utils.ridders_diff(pn,x)
dpndx_analytic = lambda x: A*4*np.pi*(np.exp(-(x/b)**c)*(((a-1)*b**(3-a)*x**(a-2))-(c*b**(2-a)*x**(a-1)*(x/b)**(c-1))))

plt.plot(x,dpndx)
plt.plot(x,dpndx_analytic(x),color='r',linestyle='-.')
plt.axhline(y=0)
plt.show()

root = float(utils.NewRaph_rootfinder(dpndx_analytic,1e-4,5))
y = float(pn(root)*2)
print('root=',root, ', y =',y)

#--- 2.g ---
