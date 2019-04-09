#NR_a1_2_main.py
import numpy as np 
import matplotlib.pyplot as plt
import NR_a1_2_utils as utils 

seed = 42
print('Original seed:',seed)
rng = utils.rng(seed)

#--- 2.a --- 
print('2.a:')
a = rng.rand_num(1,min=1.1,max=2.5)
b = rng.rand_num(1,min=0.5,max=2)
c = rng.rand_num(1,min=1.5,max=4)
f = lambda x: 4*np.pi* (x**(a-1))/(b**(a-3)) *np.exp(-(x/b)**c)
f_int = utils.romber_int(f,0,5)
A = 1/f_int 
print('A = {}; a,b,c = {},{},{}'.format(A,float(a),float(b),float(c)))

#--- 2.b --- 
print('2.b:')
xj = [10**-4,10**-2,10**-1,1,5]
n_x = lambda x: A*100*(x/b)**(a-3)*np.exp(-(x/b)**c)
n = n_x(xj)
x = np.logspace(np.log10(1e-4),np.log10(5),10000)
y = np.zeros(10000)
y_lin = utils.interpol_lin(xj,n,x,y)
plt.scatter(xj,n)
plt.plot(x,y_lin)
plt.xlim(left=10**-4,right=5)
plt.ylim(bottom=1e-4,top = 1e9)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('n(x)')
plt.title('Linear interpolation between different n(x) values')
plt.savefig('plots/2_b.png')
plt.close()
print('Saved interpolation plot to \'plots/2_b.png\'.')

#--- 2.c --- 
print('2.c:')
n = lambda x: A*100*(x/b)**(a-3)*np.exp(-(x/b)**c)
x = b
dndx = utils.ridders_diff(n,np.array([b]))
dndx_analitic = lambda x: (A*100) * (((a-3)*(x/b)**(a-4)*np.exp(-(x/b)**c))/b - ((c*np.exp(-(x/b)**c)*(x/b)**(a+c-4))/b)) 
dndx_an = dndx_analitic(x)
print('dn/dx at x = b: analytic = {0:.12f}; numerical = {1:.12f}'.format(float(dndx_an),float(dndx)))

#--- 2.d --- 
print('2.d:')
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
#ax = plt.figure().add_subplot(111,projection='3d')
#ax.scatter(x,y,z)
#plt.show()
print()
print('r,phi,theta:')
for i in range(len(r)):
    print(r[i],phi[i],theta[i])


#--- 2.e --- 
print('2.e:')
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
plt.savefig('plots/2_e.png')
plt.close()
print('Saved histogram to \'plots/2_e.jpg\'.')

#--- 2.f --- 
print('2.f:')
dpndx = utils.ridders_diff(pn,x)
dpndx_analytic = lambda x: A*4*np.pi*(np.exp(-(x/b)**c)*(((a-1)*b**(3-a)*x**(a-2))-(c*b**(2-a)*x**(a-1)*(x/b)**(c-1))))

dpndx_0 = float(utils.NewRaph_rootfinder(dpndx_analytic,1e-4,1,rng))
new_floor = float(pn(dpndx_0)/2)
pn_new_floor = lambda x: pn(x) - new_floor
root1 = float(utils.NewRaph_rootfinder(pn_new_floor,1e-4,dpndx_0,rng))
root2 = float(utils.NewRaph_rootfinder(pn_new_floor,dpndx_0,5,rng))
print('Roots:', root1,root2)

#--- 2.g ---
print('2.g:')
counts = np.zeros((len(bins)-1))
for i in r: 
    for j in range(len(bins)-1):
        if i < bins[j+1] and i > bins[j]:
            counts[j] += 1
r_list = []
r_halo_distrib = np.zeros((1000))
        
for i in range(len(r)):
    if r[i] < bins[utils.arg_max(counts)+1] and r[i] > bins[utils.arg_max(counts)]:
        r_list.append(r[i])
        r_halo_distrib[i//100] += 1
mean = sum(r_halo_distrib)/len(r_halo_distrib)
halo_bins = np.linspace(10,45,36)
poissd = []

for i in range(len(halo_bins)):
    poissd.append(utils.poisson_distribution(round(mean),int(halo_bins[i])))

plt.hist(r_halo_distrib,halo_bins,density=True)
plt.plot(halo_bins,poissd)
plt.title('Number of galaxies in most populous radial bin in each halo')
plt.savefig('plots/2_g.png')
print('Saved histogram to \'plots/2_g.jpg\'.')
print('For some reason the poisson distribution does not work correctly here and I don\'t know why. It does appear that the histogram follows a fairly normal distribution, which is a good sign.')
plt.close()

sr = utils.selection_sort(r_list)

median = sr[int(len(sr)/2-0.5)]
p16th = sr[round(len(sr)*0.16)-1]
p84th = sr[round(len(sr)*0.84)-1]
print('Length: {}, median: {}, 16th: {}, 84th: {}'.format(len(sr),median,p16th,p84th))

#--- 2.h --- 
print('2.h:')
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
print('Interpolator was tested with following values:')
print('For a = {}, b = {}, c = {}, the interpolator returned A = {}.'.format(2.05,1.05,3.05,interpol))