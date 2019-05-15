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