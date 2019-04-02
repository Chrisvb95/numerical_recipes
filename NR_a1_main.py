#NR_a1_main.py

import numpy as np 
import matplotlib.pyplot as plt 
import sys
import NR_a1_utils as utils 
from importlib import reload
reload(utils)

seed = 42 
print('Original seed:',42)

# Poisson distribution
a1 = [[1,0],[5,10],[3,20],[2.6,40]]#,[101,200]]
for i in range(len(a1)):
	print(utils.poisson_distribution(a1[i][0],a1[i][1]))
a1 = [[1,0],[5,10],[3,20],[2.6,40],[101,200]]
#a1 = [[101,200]]
print('I hate my life')
for i in range(len(a1)):
	print(utils.poisson_distribution_new(a1[i][0],a1[i][1]))

# RNG 
rng = utils.rng(seed)
# Scatter plot
n = 1000 
rand = rng.rand_num(n)
#plt.scatter(rand[:(len(rand)-1)],rand[1:])
#plt.title('Sequential number plot for {} random numbers with seed {}'.format(1000,seed))
#plt.show()
# Histogram
n = 1000000
rand = rng.rand_num(n)
#plt.hist(rand,bins=20,range=(0,1))
#plt.show()