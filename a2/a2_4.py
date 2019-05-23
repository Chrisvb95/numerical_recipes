# a2_4
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from a2_1 import rng,box_muller,romber_int,ridders_diff

def random_field_generator_zeld(N,rng,mu=0,sig=1):
    # Prepares a random field in Fourier space
    #print(f'Generating a random field with n = {n} of dimension {N}x{N} (mu = {mu})')
    ck = np.zeros((N,N),dtype=complex)
    Sx = np.zeros((N,N),dtype=complex)
    Sy = np.zeros((N,N),dtype=complex)
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
            # Drawing a random number from normal distrib
            k = (k_x**2+k_y**2)**0.5 
            if k == 0:
                k = 1
            rand = box_muller(rng.rand_num(1),rng.rand_num(1),mu,sig)
            ck[j][i] = (rand[0]*k**(-3)) - 1j*(rand[1]*k**(-3))
            Sx[j][i] = ck[j][i]*k_x*1j
            Sy[j][i] = ck[j][i]*k_y*1j
    # Setting values of points who need to equal their own conjugates
    ck[0][0] = 0
    Sx[0][0],Sy[0][0] = 0,0
    ck[0][N//2] = (ck[0][N//2].real)**2
    Sx[0][N//2] = (Sx[0][N//2].real)**2
    Sy[0][N//2] = (Sy[0][N//2].real)**2
    ck[N//2][0] = (ck[N//2][0].real)**2
    Sx[N//2][0] = (Sx[N//2][0].real)**2
    Sy[N//2][0] = (Sy[N//2][0].real)**2
    ck[N//2][N//2] = (ck[N//2][N//2].real)**2
    Sx[N//2][N//2] = (Sx[N//2][N//2].real)**2
    Sy[N//2][N//2] = (Sy[N//2][N//2].real)**2
    # Setting values of bottom half of the field using conjugates
    for j in range((N//2)+1):
        for i in range(N):
            ck[-j][-i]= ck[j][i].conjugate()
            Sx[-j][-i]= Sx[j][i].conjugate()
            Sy[-j][-i]= Sy[j][i].conjugate()
    return Sx,Sy
#end random_field generator()

if __name__ == '__main__':
    print('--- Exercise 4 ---')
    seed = 627310980
    rng = rng(seed)
    print('Original seed:',seed)
    # --- 4.a ---
    # Setting the constants
    omega_m = 0.3
    omega_lambda = 0.7
    H0 = 70 # km/s/Mpc
    # Setting the functions
    H = lambda a : (H0**2*(omega_m*(a)**(-3)+omega_lambda))**0.5
    D_prefactor = lambda a : (5*omega_m*H0**2)/2*H(a)
    D_int = lambda a: 1/(a*H(a))**3
    a = 1/51
    D = lambda a: D_prefactor(a) * romber_int(D_int,1e-12,a)
    print(f'The linear growth factor at z = 50 (a = 1/51) is equal to: {D(a)}')
    
    # --- 4.b --- 
    # Setting the functions
    dDdt = lambda a: romber_int(D_int,1e-12,a)*5/2*omega_m*H0**4*(-2*omega_m*a**(-3)+omega_lambda)
    dDdt_analytic = dDdt(a)
    dDdt_numerical = ridders_diff(D,np.array([a]))
    print(f' The analytical value of time derivative of D(z) at z = 50 : {dDdt_analytic}')
    print(f' The numerical value of time derivative of D(z) at z = 50 : {dDdt_numerical}')

    # --- 4.c --- 
    N = 64
    # Generating S for the x and y dimensions in the Fourier plane
    Sx,Sy = random_field_generator_zeld(N,rng)
    Sx = np.fft.ifft2(Sx).real*N**1.5
    Sy = np.fft.ifft2(Sy).real*N**1.5
    # Setting the starting coordinates
    q = np.zeros((N,N,2))
    for i in range(len(q)):
        for j in range(len(q)):
            q[i][j] = i,j
    # Calculating different D(a) values in advance
    a = np.linspace(0.0025,1,90)
    Da = np.zeros(len(a))
    for i in range(len(a)):
        Da[i] = D(a[i])
    # Preparing x
    x = np.zeros((N,N,2))
    # Generating all of the frames for the movie
    for k in tqdm(range(0,90)):
        for i in range(N):
            for j in range(N):
                x[i][j][0] = (q[i][j][0]+Da[k]*Sx[i][j])%N
                x[i][j][1] = (q[i][j][1]+Da[k]*Sy[i][j])%
                
        plt.scatter(x[:,:,0],x[:,:,1],marker='.')
        plt.savefig('./plots/snap%04d.png'%k)
        plt.close()
    


