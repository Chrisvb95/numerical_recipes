import numpy as np
import NR_a2_1_utils as utils
from matplotlib import pyplot as plt
from scipy import stats
import os

seed = 42
print('Original seed:',seed)

#--- 1.a ---
print('--- Running code for exercise 1 and 2 ---')
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
plt.plot(n,P,label='Self written')
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
plt.plot(n,kuip_P,label='Self written')
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
random_num = np.genfromtxt(filename,delimiter=' ',skip_footer=1)

n = np.logspace(np.log10(10),np.log10(len(random_num)),dtype=int)
test_P,test_D = np.zeros((10,len(n)),dtype=list),np.zeros((10,len(n)),dtype=list)

for i in range(10):
    for j in range(len(n)):
        rand = np.array(random_num[:n[j],i])
        test_D[i][j],test_P[i][j] = utils.KS_Kuip_test(rand,gauss,mu,sig,Kuip=True)
   

plt.plot(n,kuip_P_s,label='Scipy (KS)',color = 'g')
for i in range(10):
    plt.plot(n,test_P[i],label = i)
plt.title('Scipy KS-Test and self-written Kuiper-Test')
plt.ylabel('$P(z)$')
plt.xlabel('N')
plt.xscale('log')
plt.legend(loc=2, bbox_to_anchor=(1,1))
plt.savefig('plots/1g.png')
plt.close()
print('Saving plots/1g.png')

#---2---
# Making initial density fields for different n values
N = 1024
df1 = utils.random_field_generator(-1,N,rng)
df1_inft = np.fft.ifft2(df1)
df2 = utils.random_field_generator(-2,N,rng)
df2_inft = np.fft.ifft2(df2)
df3 = utils.random_field_generator(-3,N,rng)
df3_inft = np.fft.ifft2(df3)
# Plotting fields
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2, 3,sharex='col', sharey='row')
ax1.imshow(np.abs(df1))
ax1.set(title='n = -1')
ax1.set(ylabel='Generated field')
ax2.imshow(np.abs(df2))
ax2.set(title='n = -2')
ax3.imshow(np.abs(df3))
ax3.set(title='n = -3')
ax4.imshow(np.abs(df1_inft))
ax4.set(ylabel='Inv. FT')
ax5.imshow(np.abs(df2_inft ))
ax6.imshow(np.abs(df3_inft ))
fig.suptitle('Initial density fields for different n',y=1.02)
fig.tight_layout()
plt.savefig('plots/2.png')
plt.close()
print('Saving plots/2.png')


# TO DO - 
# 1. Say something about the last plot
# 2. Set size to megaparsec 