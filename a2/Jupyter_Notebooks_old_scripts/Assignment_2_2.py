import numpy as np
from matplotlib import pyplot as plt

np.random.seed(42)
n = -1
N = 100
df = np.zeros((N,N),dtype=complex)

for j in range((N//2)):
    # Determining the value of k_y 
    k_y = j*2*np.pi/N
    for i in range(N):
        # Determining the value of k_x and sigma_x
        if i <= (N/2):
            k_x = (i)*2*np.pi/N
        else:
            k_x = (-N+i)*2*np.pi/N
        # Avoid dividing by 0
        if i != 0 or j != 0:
            sig = ((k_x**2+k_y**2)**0.5)**(n/2)
        else: 
            sig = 0
        # Drawing a random number from normal distrib 
        df[j][i] = np.random.normal(0,sig)+ 1j*np.random.normal(0,sig)

df[0][0] = 0
df[0][N//2] = (df[0][N//2].real)**2
df[N//2][0] = (df[N//2][0].real)**2
df[N//2][N//2] = (df[N//2][N//2].real)**2

for j in range(N//2+1):
    for i in range(N):
        df[-j][-i]= np.conj(df[j][i])

plt.imshow((df.real))
plt.show()
df = np.fft.ifft(df)
plt.imshow((df.real))
plt.show()
print(df.imag)
plt.imshow(df.imag)