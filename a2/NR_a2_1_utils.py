import numpy as np
import sys

class rng(object):
    # Rng object that is initiated with a give seed
    a1,a2,a3 = 21,35,4
    a = 4294957665
    

    def __init__(self, seed):
        self.state = np.int64(seed) 
        
    def MWC(self):
        # Multiply with carry generator
        x = self.state
        self.state = self.a*(x&(2**32-1))+(x>>32)        

    def XOR_shift(self):
        # XOR-shift generator
        x = self.state 
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
    else: 
        D = np.max((Dmin,Dmax))
    # Calculate the probability
    z = (N**0.5+0.12+0.11*N**(-0.5))*D
    if z < 1.18:
        P = (2*np.pi)**0.5*((np.exp(-1*np.pi**2/(8*z**2)))+(np.exp(-1*np.pi**2/(8*z**2)))**9+(np.exp(-1*np.pi**2/(8*z**2)))**25)
        return D,P 
    else:
        P = 1-2*((np.exp(-2*z**2))-(np.exp(-2*z**2))**4+(np.exp(-2*z**2))**9)
        return D,P
#end KS_test()