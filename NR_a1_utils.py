#NR_a1_utils.py
import numpy as np
import sys
import math # Only used for math.isnan() in NewRaph_rootfinder()

# --- Functions used in NR_a1_main --- 

def poisson_distribution(mean,k):
#Returns probability for given k and mean
	fact = np.float64(1)
	for i in range(round(k)):
		fact = fact*(k-1)
	return (mean**k*np.exp(-mean))/fact
#end poisson_distribution()
# TO DO: Allow it to handle fact > sys.maxsize


def poisson_distribution_new(mean,k):
#Returns probability for given k and mean	
	fact = 1
	mag = 0
	prev_n = 0
	for i in range(round(k)):
		temp = str(fact*(k-i))
		n_zeros = len(temp)-1-prev_n	 
		fact = int(temp[:10])
		mag += n_zeros
		prev_n = len(temp[:10])-1
		
	a = str(mean**k)
	b = str(np.exp(mean))

	x = (float(a[:10])/1e9 * 1/float(b[:10])/(fact/10**(prev_n-1)))
	mag_tot = -int(np.log10(float(b)))+len(a)-mag 
	return x*10**mag_tot
# Only works for massive numbers.......
#end poisson_distribution()
# TO DO: Allow it to handle fact > sys.maxsize

class rng(object):
# rng object that is initiated with a give seed
	def __init__(self, seed):
		self.state = np.int64(seed) 

	def LCG_gen(self):
	#Linear Congruential generator
		x = self.state 
		a,c,n = 2**32,1664525,1013904223
		self.state = x 
	#end LCG_gen()

	def XOR_shift(self):
	# XOR-shift generator
		x = self.state 
		a1,a2,a3 = 21,35,4
		x = x ^ x >> a1
		x = x ^ x << a2
		x = x ^ x >> a3
		self.state = x
	#end XOR-shift()

	def rand_num(self,l,min=0,max=1):
	# Generates 'l' random numbers between min and max
		output = []
		for i in range(l):
			self.LCG_gen()
			self.XOR_shift()
			self.LCG_gen()
			self.XOR_shift()
			output.append(self.state)
		output = np.array(output)/sys.maxsize
		return min+(output*(max-min))
	#end rand_num()
#end rng()

def central_diff(f,h,x):
# Calculates the central difference\n",
    return (f(x+h)-f(x-h))/(2*h) 
#end central_diff()

def ridders_diff(f,x):
# Differentiates using Ridder's method
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
        #print(D[i])
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

def rejection_sampler(n,p,max_x,max_y,rng):
# Rejection sampler that uses max_y value as g
	sample_x = []
	sample_y = []
	n_accpt = 0
	while n_accpt < n:
		pot_x = np.float(rng.rand_num(1,max=max_x))
		pot_y = np.float(rng.rand_num(1,max=max_y))
		if pot_y <= p(pot_x):
			sample_x.append(pot_x)
			sample_y.append(pot_y)
			n_accpt += 1
	return sample_x,sample_y
#end rejection_sampler

def secant_rootfinder(f,a,b):
# Secant method root-finder
    it = 0
    while np.abs(b-a) > 0.00001: 
        it += 1
        c = -(b-a)/float((f(b)-f(a)))*f(a)+a
        a = b
        b = c
    return a
#end secant

def falspos_rootfinder(f,a,b):
# False position method root-finder
    it = 0
    c = 0
    while np.abs(b-a) > 0.001: 
        it += 1
        c = -(b-a)/float((f(b)-f(a)))*f(a)+a
        print('a',a,', b',b)
        print('c',c)
        if f(a)*f(c) > 0:
            a = c
        else:
            b = c
    return c,it 
#end falspos_rootfiner()
					
def NewRaph_rootfinder(f,a,b,rng):
# Finds root in given range (a,b) for f 
    x = rng.rand_num(1,min=a,max=b)
    x_new = sys.maxsize
    for i in range(1000):
        f_deriv = ridders_diff(f,np.array([x]))
        if f_deriv == 0 or math.isnan(f_deriv):
            x = rng.rand_num(1,min=a,max=b)
            i = 0
            continue
        x_new = x - f(x)/f_deriv
        if abs(x_new-x) < 1e-12:
            return x_new
        else:
            x = x_new 
    return x 
#end NewRaph_rootfinder()

def selection_sort(l):
# Ascending sorting of l using selection sort
    sl = []
    for i in range(len(l)):
        min = arg_min(l)
        sl.append(l[min])
        l = l[:min]+l[(min+1):]
    return sl
#end selection_sort()

def A_calc(a,b,c):
# Calculates A with given values of a,b,c
    f = lambda x: 4*np.pi* (x**(a-1))/(b**(a-3)) *np.exp(-(x/b)**c)
    f_int = romber_int(f,0,5)
    return 1/f_int
#end A_calc()

def trilinear_interpolator(al,bl,cl,Al,x,y,z):
# Returns an interpolated value for Al based on a,b,c values
    def bracket_finder(xl,x):
        for i in range(len(xl)):
            if xl[i] > x:
                return xl[i-1],xl[i],i
        print('Could not find a backet, returning nan.')
        return float('nan'),float('nan')

    a0,a1,ai = bracket_finder(al,x)
    b0,b1,bi = bracket_finder(bl,y)
    c0,c1,ci = bracket_finder(cl,z)

    print(a0,a1,ai)
    print(b0,b1,bi)
    print(c0,c1,ci)

    xd = (x-a0)/(a1-a0)
    yd = (y-b0)/(b1-b0)
    zd = (y-c0)/(c1-c0)

    c00 = Al[ai-1][bi-1][ci-1]*(1-xd)+Al[ai][bi-1][ci-1]*xd
    c01 = Al[ai-1][bi-1][ci]*(1-xd)+Al[ai][bi-1][ci]*xd
    c10 = Al[ai-1][bi][ci-1]*(1-xd)+Al[ai][bi][ci-1]*xd
    c11 = Al[ai-1][bi][ci]*(1-xd)+Al[ai][bi][ci]*xd

    c0 = c00*(1-yd)+c10*yd
    c1 = c01*(1-yd)+c11*yd

    return c0*(1-zd)+c1*zd
#end trilinear_interpolator()

# --- Simple supporting functions --- 

def min(l):
    min = 2**64
    for i in l:
        if i < min:
            min = i
    return min
#end min()

def arg_min(l):
    min = 2**64
    arg = None
    for i in range(len(l)):
        if l[i] < min:
            min = l[i]
            arg = i
    return arg
#end arg_min()

def max(l):
    max = -2**64
    for i in l:
        if i > max:
            max = i
    return max
#end max()

def arg_max(l):
    max = -2**64
    arg = None
    for i in range(len(l)):
        if l[i] > max:
            max = l[i]
            arg = i
    return arg        
#end arg_max()