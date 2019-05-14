import numpy as np
import matplotlib.pyplot as plt

# Exercise 2 and 3 from tutorial 1

#2.d Add proposes changes and is the step that precedes commit. They are then commited to the head. In order to then send these to the remote repository one then has to push. One can decided for themselves which branch they want to push the changes to. 

# 2.i Branches are used to develop features independent of each other. One can change to a different branch using checkout -b feature_x. Rest is self explanatory. Check out http://rogerdudler.github.io/git-guide/ for more info. 

xj = np.linspace(1,20,7)
yj = 2*xj*np.sin(0.8*xj)
x = np.linspace(1,25,100)
y = np.zeros(100)

def interpol_lin(xj,yj,x,y):
	j=0
	for i in range(len(x)):	
		if x[i]>xj[j]:
			j+=1
		if j>4:
			return y		
		y[i]=(((yj[j]-yj[j-1])/(xj[j]-xj[j-1]))*(x[i]-xj[j]))+yj[j]

def nevils_alg(x,xj,yj,i,j):
	if i == j:
		return yj[i]
	else:
		return ((x-xj[j])*nevils_alg(x,xj,yj,i,j-1)-(x-xj[i])*nevils_alg(x,xj,yj,i+1,j))/(xj[i]-xj[j])
		
def interpol_neville(xj,yj,x,y):
	for z in range(len(x)):
		y[z] = nevils_alg(x[z],xj,yj,0,len(xj)-1)
	return y

# Linear interpolation
y_lin = interpol_lin(xj,yj,x,y)
#y_nev = interpol_neville(xj,yj,x,y)
plt.scatter(xj,yj)
plt.plot(x,y_lin)
plt.plot(x,y_nev)
plt.title('Linear interpol and Neville\'s algorithm')
plt.ylim(-40,40)
plt.show()