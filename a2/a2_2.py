# a2_2
import numpy as np
import sys
from matplotlib import pyplot as plt
from scipy import stats
import os
from a2_1 import rng,box_muller

# --- Functions and classes ---

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

# --- Commands, prints and plots ---
if __name__ == '__main__':
    print('--- Exercise 2 ---')
    seed = 627310980
    rng = rng(seed)
    print('Original seed:',seed)

    # Making initial density fields for different n values
    N = 1024
    df1 = random_field_generator(-1,N,rng)
    df1_inft = np.fft.ifft2(df1)
    df2 = random_field_generator(-2,N,rng)
    df2_inft = np.fft.ifft2(df2)
    df3 = random_field_generator(-3,N,rng)
    df3_inft = np.fft.ifft2(df3)
    # Plotting fields
    fig, ((ax1,ax2,ax3)) = plt.subplots(1, 3,sharex='col', sharey='row',figsize=(15,15))
    ax1.set_title('n = -1',size=18)
    ax1.imshow(np.abs(df1_inft))
    ax1.set_ylabel('Mpc',size=18)
    ax1.invert_yaxis()
    ax2.set_title('n = -2',size=18)
    ax2.imshow(np.abs(df2_inft ))
    ax3.set_title('n = -3',size=18)
    ax3.imshow(np.abs(df3_inft ))
    fig.suptitle('Initial density fields for different n',y=0.7,size=20)
    fig.tight_layout()
    plt.savefig('plots/2.png',bbox_inches='tight',pad_inches = 0)
    plt.close()
    print('Generated plots/2.png')