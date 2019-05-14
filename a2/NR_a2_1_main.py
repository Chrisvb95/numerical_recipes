import numpy as np
import NR_a2_1_utils as utils
from matplotlib import pyplot as plt
from scipy import stats
import os

seed = 42
print('Original seed:',seed)

#--- 1.a ---
# MWC and XOR-Shift
rng = utils.rng(seed)
# Scatter plot
N = 1000 
rand = rng.rand_num(N)
plt.scatter(rand[:(len(rand)-1)],rand[1:])
plt.title('Sequential number plot for {} random numbers with seed {}'.format(1000,seed))
plt.savefig('plots/1a.png')
plt.close()
print('Saving plots/1a.png')

plt.scatter(np.arange(0,N,1),rand)
plt.title('Index to number plot for {} random numbers with seed {}'.format(1000,seed))
plt.xlabel('N')
plt.ylabel('Generated value')
plt.savefig('plots/1b.png')
plt.close()
print('Saving plots/1b.png')

# Histogram
N = 1000000
rand = rng.rand_num(N)
plt.hist(rand,bins=20,range=(0,1))
plt.title('Histogram of 1,000,000 randomly generated numbers'.format(1000,seed))
plt.xlabel('Number of numbers in bin')
plt.ylabel('Number values')
plt.savefig('plots/1c.png')
plt.close()
print('Saving plots/1c.png')

#--- 1.b --- 
# Box-Muller method
N = 1000
mu, sig = 3,2.4
rand = utils.box_muller(rng.rand_num(N),rng.rand_num(N),mu,sig)
gauss = lambda x,mu,sig : 1/(2*np.pi*sig**2)**0.5*np.exp(-0.5*(x-mu)**2/sig**2)
x = np.linspace(mu-(sig*5),mu+(sig*5),1000)
plt.hist(rand[0],bins=20,label='RNG numbers',density=1)
plt.plot(x,gauss(x,mu,sig),label='Gaussian distribution')
plt.title('Histogram of {} normally-distributied random numbers'.format(1000))
plt.xlabel('Number of numbers in bin')
plt.ylabel('Number values')
plt.axvline(x=mu+sig,label='$1\sigma$',color='c',linestyle='--')
plt.axvline(x=mu+2*sig,label='$2\sigma$',color='m',linestyle='--')
plt.axvline(x=mu+3*sig,label='$3\sigma$',color='y',linestyle='--')
plt.axvline(x=mu+4*sig,label='$4\sigma$',color='k',linestyle='--')
plt.legend(frameon=False)
plt.savefig('plots/1d.png')
plt.close()
print('Saving plots/1d.png')

#--- 1.c. --- 
# KS-test
# Setting parameters
mu,sig = 0,1
rand = utils.box_muller(rng.rand_num(N),rng.rand_num(N),mu,sig)
gauss = lambda x : 1/(2*np.pi*sig**2)**0.5*np.exp(-0.5*(x-mu)**2/sig**2)
n = np.logspace(np.log10(10),np.log10(1000),dtype=int)
# Preparing arrays
P,P_s = np.zeros(len(n)),np.zeros(len(n))
d,d_s = np.zeros(len(n)),np.zeros(len(n))
# Running test for different values of N
for i in range(len(n)):
    rand = utils.box_muller(rng.rand_num(n[i]),rng.rand_num(n[i]),mu,sig)
    d[i],P[i] = utils.KS_Kuip_test(rand[0],gauss,mu,sig)
    d_s[i],P_s[i] = stats.kstest(rand[0],'norm')
# Plotting
plt.plot(n,P_s,label='Scipy')
plt.plot(n,1-P,label='Self written')
plt.title('KS-Test')
plt.ylabel('$P(z)$')
plt.xlabel('N')
plt.xscale('log')
plt.legend(loc = 'lower right',frameon=False)
plt.savefig('plots/1e.png')
plt.close()
print('Saving plots/1e.png')

#---1.d---
# Kuipers test
# Preparing arrays
kuip_P,kuip_P_s = np.zeros(len(n)),np.zeros(len(n)) 
kuip_d,kuip_d_s = np.zeros(len(n)),np.zeros(len(n))
# Running test for different values of N
for i in range(len(n)):
    rand = utils.box_muller(rng.rand_num(n[i]),rng.rand_num(n[i]),mu,sig)
    kuip_d[i],kuip_P[i] = utils.KS_Kuip_test(rand[0],gauss,mu,sig,Kuip=True)
    kuip_d_s[i],kuip_P_s[i] = stats.kstest(rand[0],'norm')
# Plotting
plt.plot(n,kuip_P_s,label='Scipy')
plt.plot(n,1-kuip_P,label='Self written')
plt.title('Scipy KS-Test and self-written Kuiper-Test')
plt.ylabel('$P(z)$')
plt.xlabel('N')
plt.xscale('log')
plt.legend(loc = 'upper right',frameon=False)
plt.savefig('plots/1f.png')
plt.close()
print('Saving plots/1f.png')

#---1.e--- 
filename = 'randomnumbers.txt'
url = 'https://home.strw.leidenuniv.nl/~nobels/coursedata/'
if not os.path.isfile(filename):
    print(f'File not found, downloading {filename}')
    os.system('wget '+url+filename)
