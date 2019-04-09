#function graveyard

def S_calc(i,j,S,S_new):
        return (4**(j-1) * S_new[j-1] - S[j-1]) / (4**(j-1)- 1)
def h_calc(a,b,n):
    for i in range(len(x)):
        return 1/(2**n-1)*(b-a) 
def comp_trapezoid(f,x,a,b,n):
    h = h_calc(x,n)
    sum = 0
    for i in range(1,n):
            sum += f(x+i*h)
    return (h/2.)*(f(a)+sum+f(b))
    
def romber_int(f,x,a,b):
    #Calculate S(0,0)
    #n = 1  
    #S = np.zeros((1,len(x)))
    #S = 0
    for n in range(1,5):
        #n += 1
        S_new = np.zeros((n,len(x)))
        S_new[0] = comp_trapezoid(f,x,n)
        #print(S_new[0])
        #print(f(x))
        for j in range(1,n):
            S_new[j] = S_calc(n,j,S,S_new)
            #print(S_new)
        S = S_new
    print(S[0])

    return S[0]

def central_diff(f,h,x):
# Calculates the central difference\n",
    return (f(x+h)-f(x-h))/(2*h) 
#end central_diff()

def ridders_differentiator(f,h,x,m):
# Differentiates using Ridder's method
    def D_calc(i,j,d,D,D_new):
    # Calculates D
        return (d**(2*(j+1)) * D[j-1] - D_new[j-1]) / (d**(2*(j+1))- 1)
    #end D_calc()
    D = np.zeros((m,200))
    d = 2
    for i in range(m):
        D_new = D      
        for j in range(i+1):    
            if j == 0:
                D_new[j] = central_diff(f,h,x)
            else:
                D_new[j] = D_calc(i,j,d,D,D_new)  
        D = D_new    
        h = h/d
        #print(D)
    return D[m-1]
    # Could still look into iterating until a certain accuracy is reached
#end ridders_differentiator()