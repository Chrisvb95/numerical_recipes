#NR_a1_3_utils.py
import numpy as np
import sys
from NR_a1_1_utils import poisson_distribution, rng, min, max, arg_max,arg_min
from NR_a1_2_utils import romber_int

def selection_sort4simplex(xs,fxs):
# Ascending sorting of xs and fxs based on fxs values using selection sort
    sfxs = []
    sxs = []
    index = []
    #print(fxs) 
    for i in range(len(fxs)):
        min = arg_min(fxs)
        sfxs.append(fxs[min])
        sxs.append(xs[min])
        #print(sxs,sfxs)
        if sfxs[i] == fxs[0]:
            fxs = fxs[(min+1):]
            xs = xs[(min+1):]
        elif sfxs[i] == fxs[-1]:
            fxs = fxs[:min]         
            xs = xs[:min]           
        else:
           #print('min',min)
            #print('xs',xs[:min],xs[(min+1):])
            fxs = np.append(fxs[:min],fxs[(min+1):])
            xs = np.concatenate((xs[:min],xs[(min+1):]))
        #print(xs,fxs)
    return np.array(sxs),np.array(sfxs)
#end selection_sort4simplex()

def downhill_simplex(f,a,b,c,la = [-2**32,2**32],lb = [-2**32,2**32],lc = [-2**32,2**32]):
#Optimizing function
    alpha,beta,gamma = 1,1,0.3
    rngen = rng(14)

    # Generate starting points
    xs = [[a,b,c]]
    for i in range(3):
        xs.append([np.round(float(a+rngen.rand_num(1)),2),np.round(float(b+rngen.rand_num(1)),2),np.round(float(c+rngen.rand_num(1)),2)])
        xs.append([float(a+rngen.rand_num(1)),float(b+rngen.rand_num(1)),float(c+rngen.rand_num(1))])
    xs = np.array(xs)

    #Centroid function
    centroid = lambda xs: np.array([sum(xs[:,0]),sum(xs[:,1]),sum(xs[:,2])])/len(xs)
    it = 0
    N = 1000
    while it < N:
        it += 1 
        # Reorder the points from best to worst 
        fxs = f(xs)
        xs,fxs = selection_sort4simplex(xs,fxs)
        #print('-------------------')
        #print('xs:',xs)
        #print('fxs:',fxs)
        #Generate a trial point by reflection 
        c = centroid(xs)
        #print('Centroid:',c)
        x_new = np.array([c + alpha*(c-xs[-1])])
        #print('x_new:',x_new)
        if x_new[0][0] < la[0] or x_new[0][0] > la[1] or x_new[0][1] < lb[0] or x_new[0][1] > lb[1] or x_new[0][2] < lc[0] or x_new[0][2] > lc[1]: 
            fx_new = 2**32 
        else:
            fx_new = f(x_new)
        #print('fx_new',fx_new) 

        if fx_new < fxs[0]: #Expansion
            #print('Expanding')
            x_exp = x_new + beta*(x_new-c)
            if f(x_exp) < f(x_new):
                xs[-1] = x_exp
            else:
                xs[-1] = x_new

        elif fx_new > fxs[-1]: #Contraction
            fx_cont = sys.maxsize
            wit = 0
            while fx_cont > fxs[-1] and wit < 10:
                wit += 1
                #print('Contracting')
                x_cont = np.array([c +gamma*(xs[-1]-c)])
                if x_cont[0][0] < la[0] or x_cont[0][0] > la[1] or x_cont[0][1] < lb[0] or x_cont[0][1] > lb[1] or x_cont[0][2] < lc[0] or x_cont[0][2] > lc[1]: 
                    fx_cont = 2**32
                    gamma = gamma*0.9 
                else:
                    fx_cont = f(x_cont) 
            xs[-1] = x_cont

        else: #Back to reflection
            #print('Back to top')
            xs[-1] = x_new
            
        if np.abs(fxs[0]-fxs[1]) < 1e-15:
            print('Found minimum after {} iterations'.format(it))
            print(xs[0],fxs[0])
            return xs[0],fxs[0]
    print('Reached maximum number of iterations, returning best coordinates...')
    return xs[0],fxs[0]
#end downhill_simplex()

def minlog_likelyhood(abc):
# Returns the log_likelyhood of n(x) for given parameters
    a,b,c = abc[:,0],abc[:,1],abc[:,2]

    #Calculating A
    A = []
    for i in range(len(abc)):
        f = lambda x: 4*np.pi* (x**(a[i]-1))/(b[i]**(a[i]-3)) *np.exp(-(x/b[i])**c[i])
        A.append(1/romber_int(f,0,5))
    print(A)

    #Calculating rest of the formula
    N = len(data)
    logSum,expSum = 0,0
    for i in data:
        logSum += np.log(i)
        expSum += -1*(i/b)**c 
    return -(N*np.log(A) + (a-1)*(logSum-N*np.log(b)) + expSum)
#end log_likelyhood()