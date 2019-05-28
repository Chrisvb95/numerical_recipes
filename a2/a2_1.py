# a2_1
import numpy as np
import sys
from matplotlib import pyplot as plt
from scipy import stats
import os
from astropy.stats import kuiper

# --- Functions and classes ---

class rng(object):
    # Rng object that is initiated with a give seed
    a1,a2,a3 = np.int64(21),np.int64(35),np.int64(4)
    a = 4294957665
    

    def __init__(self, seed):
        self.state = np.int64(seed) 
        
    def MWC(self):
        # Multiply with carry generator
        x = np.int64(self.state)
        self.state = self.a*(x&(2**32-1))+(x>>32)        

    def XOR_shift(self):
        # XOR-shift generator
        x = np.int64(self.state) 
        x = x ^ x >> self.a1
        x = x ^ x << self.a2
        x = x ^ x >> self.a3
        self.state = np.int64(x)
    #end XOR-shift()

    def rand_num(self,l,min=0,max=1):
        # Generates 'l' random numbers between min and max
        output = []
        for i in range(l):
            self.XOR_shift()
            self.MWC()
            self.XOR_shift()
            output.append(self.state)
        output = np.array(output)/sys.maxsize
        return min+(output*(max-min))
    #end rand_num()
#end rng()

def box_muller(u1,u2,mu,sigma):
    # Implementation of the Box Muller transform
    x1 = (-2*np.log(u1))**0.5*np.sin(2*np.pi*u2)
    x2 = (-2*np.log(u1))**0.5*np.cos(2*np.pi*u2)
    return x1*sigma+mu,x2*sigma+mu
#end box_muller

def central_diff(f,h,x):
    # Calculates the central difference\n",
    return (f(x+h)-f(x-h))/(2*h) 
#end central_diff()

def ridders_diff(f,x):
    #Differentiates using Ridder's method
    m = 10
    D = np.zeros((m,len(x)))
    d = 2
    h = 0.001
    for i in range(m):
        D_new = D      
        for j in range(i+1):    
            if j == 0:
                D_new[j] = central_diff(f,h,x)
            else:
                D_new[j] = (d**(2*(j+1))*D[j-1]-D_new[j-1])/(d**(2*(j+1))-1)  
        D = D_new    
        h = h/d
    return D[m-1]
#end ridders_diff()

def comp_trapezoid(f,a,b,n):
    # Composite trapezoid rule used in romber_int()
    h = 1/(2**(n-1))*(b-a)
    sum = 0
    for i in range(1,2**(n-1)):
        sum += f(a+i*h)
    return (h/2.)*(f(a)+2*sum+f(b))
#end comp_trapezoid()

def romber_int(f,a,b):
    # Integrates from a to b up to an accuracy of 6 decimals
    for n in range(1,10):
        S_new = np.zeros((n))
        S_new[0] = comp_trapezoid(f,a,b,n)
        for j in range(2,n+1):
            S_new[j-1] = (4**(j-1)*S_new[j-2]-S[j-2])/(4**(j-1)- 1)
        S = S_new
        if n > 3:
            if abs(S[-2]-S[-1]) < 1e-6:
                return S[-1]
    return S[-1]
#end romber_int()

def KS_Kuip_test(sample,f,mu,sig,Kuip=False):
    # Implementation of the Kalgorov-Smirnov test
    N = len(sample)
    x = np.linspace(mu-5*sig,mu+5*sig,1000)
    F, Fn = np.zeros(len(x)), np.zeros(len(x))
    Dmin = 0
    Dmax = 0
    for i in range(len(Fn)):
        Fn[i] = len(np.where(sample<=x[i])[0])/N
        F[i] = romber_int(f,x[0],x[i])
        Dn = F[i] - Fn[i]
        if Dn > Dmin:
            Dmin = Dn
        Dn = Fn[i] - F[i]
        if Dn > Dmax: 
            Dmax = Dn
    # Determine the manner in which D is calculated
    if Kuip:
        D = Dmin+Dmax
        # Calculate the probability
        z = (N**0.5+0.155+0.24*N**(-0.5))*D
        #print(z)
        if z < 0.4:
            P = 1
        else:
            P = 0    
            for i in range(1,1000):
                Pi = 2*(4*i**2*z**2-1)*np.exp(-2*i**2*z**2)
                P += Pi 
                if Pi <= 0.00001:
                    return D, P
        return D, P
    else: 
        D = np.max((Dmin,Dmax))
        # Calculate the probability
        z = (N**0.5+0.12+0.11*N**(-0.5))*D
        if z < 1.18:
            P = (2*np.pi)**0.5*((np.exp(-1*np.pi**2/(8*z**2)))+(np.exp(-1*np.pi**2/(8*z**2)))**9+(np.exp(-1*np.pi**2/(8*z**2)))**25)
        else:
            P = 1-2*((np.exp(-2*z**2))-(np.exp(-2*z**2))**4+(np.exp(-2*z**2))**9)
        return D,1-P
#end KS_test()

def Ks_test_2s(sample1,sample2,mu,sig,Kuip=False):
    # Implementation of the Kalgorov-Smirnov test
    N1,N2 = len(sample1),len(sample2)
    x = np.linspace(mu-5*sig,mu+5*sig,1000)
    F, G = np.zeros(len(x)), np.zeros(len(x))
    Dmin,Dmax = 0,0
    for i in range(len(x)):
        F[i] = len(sample1[sample1<=x[i]])/N1
        G[i] = len(sample2[sample2<=x[i]])/N2
        
        Dn = F[i] - G[i]
        if Dn > Dmin:
            Dmin = Dn
        Dn = G[i] - F[i]
        if Dn > Dmax: 
            Dmax = Dn
    # Determine the manner in which D is calculated
    if Kuip:
        D = Dmin+Dmax
    else: 
        D = np.max((Dmin,Dmax))
    # Calculate the probability
    z = (N1**0.5+0.12+0.11*N1**(-0.5))*D
    if z < 1.18:
        P = (2*np.pi)**0.5*((np.exp(-1*np.pi**2/(8*z**2)))+(np.exp(-1*np.pi**2/(8*z**2)))**9+(np.exp(-1*np.pi**2/(8*z**2)))**25)
        return D,1-P 
    else:
        P = 1-2*((np.exp(-2*z**2))-(np.exp(-2*z**2))**4+(np.exp(-2*z**2))**9)
        return D,1-P
#end KS_test()

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

def gauss_cdf(x):
        gauss = lambda x : 1/(2*np.pi*sig**2)**0.5*np.exp(-0.5*(x-mu)**2/sig**2)
        cdf = np.zeros(len(x))
        for i in range(len(x)):
            cdf[i] = romber_int(gauss,-5,x[i])
        return cdf

# --- Commands, prints and plots ---
if __name__ == '__main__':
    print('--- Exercise 1 ---')
    seed = 627310980
    rng = rng(seed)
    print('Original seed:',seed)

    #--- 1.a ---
    # MWC and XOR-Shift
    N = 1000 
    rand = rng.rand_num(N)
    # Sequential number plot
    plt.scatter(rand[:(len(rand)-1)],rand[1:])
    plt.title('Sequential number plot for {} random numbers with seed {}'.format(1000,seed))
    plt.savefig('plots/1a.png')
    plt.close()
    print('Generated plots/1a.png')
    # Index to number plot
    plt.scatter(np.arange(0,N,1),rand)
    plt.title('Index to number plot for {} random numbers with seed {}'.format(1000,seed))
    plt.xlabel('N')
    plt.ylabel('Generated value')
    plt.savefig('plots/1b.png')
    plt.close()
    print('Generated plots/1b.png')
    # Histogram
    N = 1000000
    rand = rng.rand_num(N)
    plt.hist(rand,bins=20,range=(0,1))
    plt.title('Histogram of 1,000,000 randomly generated numbers'.format(1000,seed))
    plt.xlabel('Number of numbers in bin')
    plt.ylabel('Number values')
    plt.savefig('plots/1c.png')
    plt.close()
    print('Generated plots/1c.png')

    #--- 1.b --- 
    # Box-Muller method
    N = 1000
    mu, sig = 3,2.4
    rand = box_muller(rng.rand_num(N),rng.rand_num(N),mu,sig)
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
    print('Generated plots/1d.png')

    #--- 1.c. --- 
    # KS-test
    # Setting parameters
    mu,sig = 0,1
    rand = box_muller(rng.rand_num(N),rng.rand_num(N),mu,sig)
    gauss = lambda x : 1/(2*np.pi*sig**2)**0.5*np.exp(-0.5*(x-mu)**2/sig**2)
    n = np.logspace(np.log10(10),np.log10(100000),dtype=int)
    # Preparing arrays
    P,P_s = np.zeros(len(n)),np.zeros(len(n))
    d,d_s = np.zeros(len(n)),np.zeros(len(n))
    # Running test for different values of N
    for i in range(len(n)):
        rand = box_muller(rng.rand_num(n[i]),rng.rand_num(n[i]),mu,sig)
        d[i],P[i] = KS_Kuip_test(rand[0],gauss,mu,sig)
        d_s[i],P_s[i] = stats.kstest(rand[0],'norm')
    # Plotting
    plt.plot(n,P_s,label='Scipy')
    plt.scatter(n,P_s)
    plt.plot(n,P,label='Self written')
    plt.scatter(n,P)
    plt.title('KS-Test')
    plt.ylabel('$P(z)$')
    plt.xlabel('N')
    plt.xscale('log')
    lgd = plt.legend(loc=2, bbox_to_anchor=(1,1))
    plt.savefig('plots/1e.png',bbox_inches='tight')
    plt.close()
    print('Generated plots/1e.png')

    #---1.d---
    # Kuipers test
    # Preparing arrays
    kuip_P,kuip_P_ast = np.zeros(len(n)),np.zeros(len(n)) 
    kuip_d,kuip_d_ast = np.zeros(len(n)),np.zeros(len(n))

    # Running test for different values of N
    rand_bm = box_muller(rng.rand_num(n[-1]),rng.rand_num(n[-1]),mu,sig)   
    for i in range(len(n)):
        rand = rand_bm[0][:n[i]]
        kuip_d[i],kuip_P[i] = KS_Kuip_test(rand,gauss,mu,sig,Kuip=True)
        kuip_d_ast[i],kuip_P_ast[i] = kuiper(rand,gauss_cdf)
    # Plotting
    plt.plot(n,kuip_P_ast,label='Astropy')
    plt.plot(n,kuip_P,label='Self written')
    plt.title('Astropy Kuiper-Test and self-written Kuiper-Test')
    plt.ylabel('$P(z)$')
    plt.xlabel('N')
    plt.xscale('log')
    lgd = plt.legend(loc=2, bbox_to_anchor=(1,1))
    plt.savefig('plots/1f.png', bbox_inches='tight')
    plt.close()
    print('Generated plots/1f.png')

    #---1.e--- 
    # Testing on given random numbers
    filename = 'randomnumbers.txt'
    url = 'https://home.strw.leidenuniv.nl/~nobels/coursedata/'
    if not os.path.isfile(filename):
        print(f'File not found, downloading {filename}')
        os.system('wget '+url+filename)
    random_num = np.genfromtxt(filename,delimiter=' ',skip_footer=1)

    n = np.logspace(np.log10(10),np.log10(len(random_num)),dtype=int)
    test_P,test_D = np.zeros((10,len(n)),dtype=list),np.zeros((10,len(n)),dtype=list)
    # Applying Kuipers test 
    for i in range(10):
        for j in range(len(n)):
            rand = np.array(random_num[:n[j],i])
            test_D[i][j],test_P[i][j] = KS_Kuip_test(rand,gauss,mu,sig,Kuip=True)
    # Plotting
    for i in range(10):
        plt.plot(n,test_P[i],label = i)
    plt.title('KS-test performed on given dataset')
    plt.ylabel('$P(z)$')
    plt.xlabel('N')
    plt.xscale('log')
    plt.legend(loc=2, bbox_to_anchor=(1,1))
    plt.savefig('plots/1g.png',bbox_inches='tight')
    plt.close()
    print('Generated plots/1g.png')