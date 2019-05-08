#NR_a1_3_main.py
import numpy as np 
import matplotlib.pyplot as plt
import NR_a1_3_utils as utils 

#--- 3.a --- 
print('3.a:')
a,b,c = 10.,3.,3.
xs = [[a,b,c]]
f = lambda xs : (xs[:,0]-1/3)**2+(xs[:,1]-2/3)**2+(xs[:,2]-1/3)**2
min = utils.downhill_simplex(f,a,b,c)
#print(min)
#print('Found the minimum value {4} at a,b,c = {0},{1},{2}')(min[0][0],min[0][1],min[0][2],min[1])

x = utils.data_unpacker('satgals_m15.txt')
la,lb,lc = [1.1,2.5],[0.5,2.],[1.5,4.] #Giving a,b,c value range
min = utils.downhill_simplex(utils.minlog_likelyhood,1.8,1.25,2.75,la,lb,lc)
#print('Found the minimum value {4} at a,b,c = {0},{1},{2}')(min[0][0],min[0][1],min[0][2],min[1])
print('This almost works correctly, only thing that needs to be done is to properly enforce range of abc, values.')