#NR_a1_1_main.py
import numpy as np 
import matplotlib.pyplot as plt 
import NR_a1_1_utils as utils 

seed = 42
print('Original seed:',seed)

#--- 1.a ---
# Poisson distribution
print('1.a:')
a1 = [[1,0],[5,10],[3,20],[2.6,40]]
print('Simple poisson distribution:')
for i in range(len(a1)):
    print('P({}) with mean {}:'.format(a1[i][1],a1[i][0]),utils.poisson_distribution(a1[i][0],a1[i][1]))
print('Large-number poisson distribution (only works for large numbers):')
a1 = [[101,200]]
for i in range(len(a1)):
    print('P({}) with mean {}:'.format(a1[i][1],a1[i][0]),utils.poisson_distribution_new(a1[i][0],a1[i][1]))

#--- 1.b ---
print('1.b:')
# RNG 
rng = utils.rng(seed)
# Scatter plot
N = 1000 
rand = rng.rand_num(N)
plt.scatter(rand[:(len(rand)-1)],rand[1:])
plt.title('Sequential number plot for {} random numbers with seed {}'.format(1000,seed))
plt.savefig('plots/1_b_1.png')

# Histogram
N = 1000000
rand = rng.rand_num(N)
plt.hist(rand,bins=20,range=(0,1))
plt.title('Histogram of 1,000,000 randomly generated numbers'.format(1000,seed))
plt.xlabel('Number of numbers in bin')
plt.ylabel('Number values')
plt.savefig('plots/1_b_2.png')
print('Saving Histogram and scatter plot.')