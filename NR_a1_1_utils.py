#NR_a1_1_utils.py
import numpy as np
import sys
import math # Only used for math.isnan() in NewRaph_rootfinder()

def poisson_distribution(mean,k):
    fact = np.float(0)
    for i in range(round(k)):
        fact += np.log(k-1)
    fact = np.exp(fact)
    return (mean**k*np.exp(-mean))/fact

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

class rng(object):
# rng object that is initiated with a give seed
    def __init__(self, seed):
        self.state = np.int64(seed) 

    def LCG_gen(self):
    #Linear Congruential generator
        x = self.state 
        a,c,m = 2**32,1664525,1013904223
        self.state = np.int64((a*x+c)%m) 
    #end LCG_gen()

    def XOR_shift(self):
    # XOR-shift generator
        x = self.state 
        a1,a2,a3 = 21,35,4
        x = x ^ x >> a1
        x = x ^ x << a2
        x = x ^ x >> a3
        self.state = np.int64(x)
    #end XOR-shift()

    def rand_num(self,l,min=0,max=1):
    # Generates 'l' random numbers between min and max
        output = []
        for i in range(l):
            self.XOR_shift()
            self.XOR_shift()
            self.LCG_gen()
            self.XOR_shift()
            self.XOR_shift()
            output.append(self.state)
        output = np.array(output)/sys.maxsize
        return min+(output*(max-min))
    #end rand_num()
#end rng()

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