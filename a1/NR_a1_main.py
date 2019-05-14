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

#--- 1.a ---
# Poisson distribution
a1 = [[1,0],[5,10],[3,20],[2.6,40]]
print('Simple poisson distribution:')
for i in range(len(a1)):
	print('P({}) with mean {}:'.format(a1[i][1],a1[i][0]),utils.poisson_distribution(a1[i][0],a1[i][1]))
print('Large-number poisson distribution (only works for large numbers)')
a1 = [[101,200]]
for i in range(len(a1)):
	print('P({}) with mean {}:'.format(a1[i][1],a1[i][0]),utils.poisson_distribution_new(a1[i][0],a1[i][1]))

#--- 1.b ---
# RNG 
rng = utils.rng(seed)
# Scatter plot
N = 1000 
rand = rng.rand_num(N)
plt.scatter(rand[:(len(rand)-1)],rand[1:])
plt.title('Sequential number plot for {} random numbers with seed {}'.format(1000,seed))
plt.savefig('images/1_b_1.png')
plt.show()

# Histogram
N = 1000000
rand = rng.rand_num(N)
plt.hist(rand,bins=20,range=(0,1))
plt.title('Histogram of 1,000,000 randomly generated numbers'.format(1000,seed))
plt.xlabel('Number of numbers in bin')
plt.ylabel('Number values')
plt.savefig('images/1_b_2.png')
plt.show()

#--- 2.a --- 
a = rng.rand_num(1,min=1.1,max=2.5)
b = rng.rand_num(1,min=0.5,max=2)
c = rng.rand_num(1,min=1.5,max=4)
f = lambda x: 4*np.pi* (x**(a-1))/(b**(a-3)) *np.exp(-(x/b)**c)
f_int = utils.romber_int(f,0,5)
A = 1/f_int 
print('A = {}; a,b,c = {},{},{}'.format(A,float(a),float(b),float(c)))

#--- 2.b --- 

xj = [10**-4,10**-2,10**-1,1,5]
n_x = lambda x: A*100*(x/b)**(a-3)*np.exp(-(x/b)**c)
n = n_x(xj)
print(n)
x = np.logspace(np.log10(1e-4),np.log10(5),10000)
y = np.zeros(10000)
y_lin = utils.interpol_lin_log(xj,n,x,y)

plt.scatter(xj,n)
plt.plot(x,y_lin)
plt.xlim(left=10**-4,right=5)
plt.ylim(bottom=1e-4,top = 1e9)
plt.xscale('log')
plt.yscale('log')
plt.show()

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
print()
print('r,phi,theta:')
for i in range(len(r)):
    print(r[i],phi[i],theta[i])


#--- 2.e --- 

N = 100000
samples = utils.rejection_sampler(N,pn,5,g,rng)
r = samples[0]
bins = np.logspace(np.log10(1e-4),np.log10(xmax),num=21)
plt.hist(r,bins=bins,density=True)
plt.plot(bins,pn(bins),label = '$N(x) = n(x)4\pi x^2 dx$')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('N(x)')
plt.xlabel('x')
plt.legend()
plt.title('Histogram of avg number of galaxies for different values of x')
plt.show()
plt.savefig('images/2_e.jpg')

#--- 2.f --- 
dpndx = utils.ridders_diff(pn,x)
dpndx_analytic = lambda x: A*4*np.pi*(np.exp(-(x/b)**c)*(((a-1)*b**(3-a)*x**(a-2))-(c*b**(2-a)*x**(a-1)*(x/b)**(c-1))))

dpndx_0 = float(utils.NewRaph_rootfinder(dpndx_analytic,1e-4,1,rng))
new_floor = float(pn(dpndx_0)/2)
pn_new_floor = lambda x: pn(x) - new_floor
root1 = float(utils.NewRaph_rootfinder(pn_new_floor,1e-4,dpndx_0,rng))
root2 = float(utils.NewRaph_rootfinder(pn_new_floor,dpndx_0,5,rng))
print('Roots:', root1,root2)

#--- 2.g ---
counts = np.zeros((len(bins)-1))
for i in r: 
    for j in range(len(bins)-1):
        if i < bins[j+1] and i > bins[j]:
            counts[j] += 1
r_list = []
for i in r:
    if i < bins[utils.arg_max(counts)+1] and i > bins[utils.arg_max(counts)]:
        r_list.append(i)
        
sr = utils.selection_sort(r_list)

for i in range(len(r)):
    if r[i] < bins[utils.arg_max(counts)+1] and r[i] > bins[utils.arg_max(counts)]:
        r_list.append(r[i])
        r_halo_distrib[i//100] += 1
mean = sum(r_halo_distrib)/len(r_halo_distrib)
halo_bins = np.linspace(10,45,36)
poissd = []

# For some reason the poisson distribution does not work correctly here and I don't know why.
for i in range(len(halo_bins)):
    poissd.append(utils.poisson_distribution(round(mean),int(halo_bins[i])))


plt.hist(r_halo_distrib,halo_bins,density=True)
plt.plot(halo_bins,poissd)
plt.title('Number of galaxies in most populous radial bin in each halo')
plt.show()
plt.savefig('images/2_g.png')

median = sr[int(len(sr)/2-0.5)]
p16th = sr[round(len(sr)*0.16)-1]
p84th = sr[round(len(sr)*0.84)-1]
print('Length: {}, median: {}, 16th: {}, 84th: {}'.format(len(sr),median,p16th,p84th))

#--- 2.h --- 
al = np.linspace(1.1,2.5,15)
bl = np.linspace(0.5,2,16)
cl = np.linspace(1.5,4,26)
param = np.array((al,bl,cl))
Al = np.zeros([len(al),len(bl),len(cl)])
for i in range(len(al)):
    for j in range(len(bl)):
        for k in range(len(cl)):
            Al[i][j][k] = utils.A_calc(al[i],bl[j],cl[k])

interpol = utils.trilinear_interpolator(al,bl,cl,Al,2.05,1.05,3.05)
print(interpol)

# --- 3.a ---
