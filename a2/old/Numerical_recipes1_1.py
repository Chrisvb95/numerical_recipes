import numpy as np

# Exercise 1 from tutorial 1 

def average(l):
	return sum(l)/len(l)

def std_dev(l):
	return np.std(l)

l = list(range(1,101))

print('Average:',average(l))
print('Std. dev:',std_dev(l))

l_odd = l[0::2]
l_even = l[1::3]

print('Average for odd:',average(l_odd))
print('Std. dev for odd:',std_dev(l_odd))

print('Average for even:',average(l_even))
print('Std. dev for evem:',std_dev(l_even))

l_big = l[:9]+l[20:44]+l[57:]
l_small = l[10:20]+l[45:57]

print('Average for big set:',average(l_big))
print('Std. dev for big set:',std_dev(l_big))

print('Average for small set:',average(l_small))
print('Std. dev for small set:',std_dev(l_small))




