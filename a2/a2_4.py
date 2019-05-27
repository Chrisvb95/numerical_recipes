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

def random_field_generator_zeld_3D(N,rng,mu=0,sig=1):
    # Prepares a random field in Fourier space
    #print(f'Generating a random field with n = {n} of dimension {N}x{N} (mu = {mu})')
    ck = np.zeros((N,N,N),dtype=complex)
    Sx = np.zeros((N,N,N),dtype=complex)
    Sy = np.zeros((N,N,N),dtype=complex)
    Sz = np.zeros((N,N,N),dtype=complex)

    for l in range(N):
        if l <= (N//2):
            k_z = (l)*2*np.pi/N
        else:
            k_z = (-N+l)*2*np.pi/N
        #print(k_z)
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
                k = (k_x**2+k_y**2+k_z**2)**0.5 
                if k == 0:
                    k = 1
                rand = box_muller(rng.rand_num(1),rng.rand_num(1),mu,sig)
                ck[l][j][i] = (rand[0]*k**(-3)) - 1j*(rand[1]*k**(-3))
                Sx[l][j][i] = ck[l][j][i]*k_x*1j
                Sy[l][j][i] = ck[l][j][i]*k_y*1j
                Sz[l][j][i] = ck[l][j][i]*k_z*1j
        # Setting values of points who need to equal their own conjugates
        ck[l][0][0] = 0
        Sx[l][0][0],Sy[l][0][0],Sz[l][0][0] = 0,0,0
        ck[l][0][N//2] = (ck[l][0][N//2].real)**2
        Sx[l][0][N//2] = (Sx[l][0][N//2].real)**2
        Sy[l][0][N//2] = (Sy[l][0][N//2].real)**2
        Sz[l][0][N//2] = (Sz[l][0][N//2].real)**2
        ck[l][N//2][0] = (ck[l][N//2][0].real)**2
        Sx[l][N//2][0] = (Sx[l][N//2][0].real)**2
        Sy[l][N//2][0] = (Sy[l][N//2][0].real)**2
        Sz[l][N//2][0] = (Sz[l][N//2][0].real)**2
        ck[l][N//2][N//2] = (ck[l][N//2][N//2].real)**2
        Sx[l][N//2][N//2] = (Sx[l][N//2][N//2].real)**2
        Sy[l][N//2][N//2] = (Sy[l][N//2][N//2].real)**2
        Sz[l][N//2][N//2] = (Sz[l][N//2][N//2].real)**2
        # Setting values of bottom half of the field using conjugates
        for j in range((N//2)+1):
            for i in range(N):
                ck[l][-j][-i]= ck[l][j][i].conjugate()
                Sx[l][-j][-i]= Sx[l][j][i].conjugate()
                Sy[l][-j][-i]= Sy[l][j][i].conjugate()
                Sz[l][-j][-i]= Sz[l][j][i].conjugate()
    return Sx,Sy,Sz
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
    H = lambda a : H0*((omega_m*(a)**(-3)+omega_lambda))**0.5
    D_prefactor = lambda a : (5*omega_m*H0**2)/2*H(a)
    dIda = lambda a: 1/(a*H(a))**3
    I = lambda a: romber_int(dIda,1e-12,a)
    a = 1/51
    D = lambda a: D_prefactor(a) * I(a)
    print(f'The linear growth factor at z = 50 (a = 1/51) is equal to: {D(a)}')
    
    # --- 4.b --- 
    # Setting the functions
    pre_fact = lambda a: 5*omega_m*H0**3/(2*a**(0.5)) 
    dHda = lambda a: -3*omega_m/(2*(a**5*(omega_m+omega_lambda*a**3))**0.5)
    dDdt = lambda a: pre_fact(a)*(dHda(a)*I(a)+dIda(a)*H(a))
    dDdt_numerical = ridders_diff(D,np.array([a]))*H0/(a)**0.5
    print(f' The analytical value of time derivative of D(z) at z = 50 : {dDdt}')
    print(f' The numerical value of time derivative of D(z) at z = 50 : {dDdt_numerical}')

    # --- 4.c --- 
    print('Starting 2D N-body simulation')
    # Preparing parameters that will be used in both simulations
    N = 64
    a = np.linspace(0.0025,1,90)
    Da = np.zeros(len(a))

    # 2D - Generating S for the x and y dimensions in the Fourier plane
    Sx,Sy = random_field_generator_zeld(N,rng)
    Sx = np.fft.ifft2(Sx).real*N
    Sy = np.fft.ifft2(Sy).real*N
    
    # Setting the starting coordinates
    q = np.zeros((N,N,2))
    for i in range(len(q)):
        for j in range(len(q)):
            q[i][j] = i,j
    
    # Preparing values and arrays for the plotting of y vs a
    da = a[1]-a[0]
    p = lambda a,S : -1*(a-da/2)**2*dDdt(a-da/2)*S
    Py = np.zeros((len(a),10))
    Xy = np.zeros((len(a),10))

    # Iterating through all the a values
    x2D = np.zeros((N,N,2))
    for k in tqdm(range(0,90)):
        # Calculating D and D*S
        Da[k] = D(a[k])
        DSx = Da[k]*Sx
        DSy = Da[k]*Sy
        # Calculating the new x positions
        x2D[:,:,0] = (q[:,:,0]+DSx)%N
        x2D[:,:,1] = (q[:,:,1]+DSy)%N
        # Saving for momentum plot
        Xy[k] = x2D[0,:10,1]
        Py[k] = p(a[k],Sy[0,:10])
        # Plotting
        plt.scatter(x2D[:,:,0],x2D[:,:,1],marker='.')
        plt.title('2D N-body simulation')
        plt.ylabel('Mpc')
        plt.xlabel(f'a = {np.round(a[k],3)}')
        plt.savefig('./plots/2Dmovie/snap%04d.png'%k)
        plt.close()

    plt.plot(a,Py)
    plt.xlabel('a')
    plt.ylabel('Momentum $p_y(a)$')
    plt.title('Momentum of the first 10 particles in y-direction')
    plt.savefig('./plots/4a.png')
    plt.close()

    plt.plot(a,Xy)
    plt.xlabel('a')
    plt.ylabel('Position $y(a)$')
    plt.title('Position of the first 10 particles in y-direction')
    plt.savefig('./plots/4b.png')
    plt.close()

    print('2D N-body simulation completed')

    # --- 4.d ---
    print('Starting 3D N-body simulation')

    # 3D - Generating S for the x and y dimensions in the Fourier plane
    Sx,Sy,Sz = random_field_generator_zeld_3D(64,rng)
    Sx = np.fft.ifftn(Sx).real*N**2
    Sy = np.fft.ifftn(Sy).real*N**2
    Sz = np.fft.ifftn(Sz).real*N**2

    # Setting the starting coordinates
    q = np.zeros((N,N,N,3))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                q[i][j][k] = i,j,k

    # Preparing arrays for recording of momentum
    Pz = np.zeros((len(a),10))
    Xz = np.zeros((len(a),10))

    # Iterating through all the a values
    x3D = np.zeros((N,N,N,3))
    for k in tqdm(range(0,90)):
        # Calculating D and D*S
        Da[k] = D(a[k])
        DSx = Da[k]*Sx
        DSy = Da[k]*Sy
        DSz = Da[k]*Sz
        # Calculating the new x positions
        x3D[:,:,:,0] = (q[:,:,:,0]+DSx)%N
        x3D[:,:,:,1] = (q[:,:,:,1]+DSy)%N
        x3D[:,:,:,2] = (q[:,:,:,2]+DSz)%N
        # Saving for momentum plot
        Xz[k] = x3D[0,0,:10,2]
        Pz[k] = p(a[k],Sz[0,0,:10])
        # Producing the slices that will be plotted
        xy = x3D[(x3D[:,:,:,2]>31.5) & (x3D[:,:,:,2]<=32.5)]
        xz = x3D[(x3D[:,:,:,1]>31.5) & (x3D[:,:,:,1]<=32.5)]
        yz = x3D[(x3D[:,:,:,0]>31.5) & (x3D[:,:,:,0]<=32.5)]
        # Plotting
        plt.scatter(xy[:,0],xy[:,1],marker='.')
        plt.title('3D N-body simulation: xy')
        plt.ylabel('Mpc')
        plt.xlabel(f'a = {np.round(a[k],3)}')
        plt.savefig('./plots/3Dmovie/xy/snap%04d.png'%k)
        plt.close()
        
        plt.scatter(xz[:,0],xz[:,2],marker='.')
        plt.title('3D N-body simulation: xz')
        plt.ylabel('Mpc')
        plt.xlabel(f'a = {np.round(a[k],3)}')
        plt.savefig('./plots/3Dmovie/xz/snap%04d.png'%k)
        plt.close()
        
        plt.scatter(yz[:,1],yz[:,2],marker='.')
        plt.title('3D N-body simulation: yz')
        plt.ylabel('Mpc')
        plt.xlabel(f'a = {np.round(a[k],3)}')
        plt.savefig('./plots/3Dmovie/yz/snap%04d.png'%k)
        plt.close()

    plt.plot(a,Pz)
    plt.xlabel('a')
    plt.ylabel('Momentum $p_z(a)$')
    plt.title('Momentum of the first 10 particles in z-direction')
    plt.savefig('./plots/4c.png')
    plt.close()

    plt.plot(a,Xz)
    plt.xlabel('a')
    plt.ylabel('Position $z(a)$')
    plt.title('Position of the first 10 particles in z-direction')
    plt.savefig('./plots/4d.png')
    plt.close()

    print('3D N-body simulation completed')