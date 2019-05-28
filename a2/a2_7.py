#a2_7.py
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
import os


class Point(): 
    def __init__(self,coordinates):
        self.x = coordinates[0]
        self.y = coordinates[1]
        #self.type = o_type 

class Node():
    def __init__(self,center,height,width,points):
        self.center = center
        self.height = height
        self.width = width
        self.points = points
        self.children = []
        
    def height(self):
        return self.height
    def width(self):
        return self.width
    def points(self):
        return self.points
    def print_all(self):
        print(f'center:{self.center},height:{self.height},width:{self.width}')
        
class Tree():
    def __init__(self,threshold,data):
        self.threshold = threshold
        
        # Make the point objects and store in list
        self.points = []
        for i in range(len(data)):
            self.points.append(Point(data[i])) 
        # Determine the dimensions of the original box 
        # We are assuming that the box is square for now
        dx = np.max(data[:,0])-np.min(data[:,0])+0.1
        dy = np.max(data[:,1])-np.min(data[:,1])+0.1
        self.d = np.max((dx,dy))
        center = (np.min(data[:,0])+self.d/2,np.min(data[:,1])+self.d/2)
        # Make the root node
        self.root = Node(center,self.d,self.d,self.points)
        # Now we are going to build the tree:
        builder(self.root,self.threshold)

    def get_points(self):
        return self.points
    
    def graph(self):
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        root = self.root
        # Root:
        c = find_children(self.root)
        for n in c:
            x0,y0 = n.center[0]-n.width/2,n.center[1]-n.height/2
            ax.add_patch(patches.Rectangle((x0,y0), n.width, n.height, fill=False))
        x = [point.x for point in self.points]
        y = [point.y for point in self.points]
        plt.scatter(x, y, marker='.')
        plt.title('Barnes Hut Quadtree')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('./plots/7.png')
        plt.close()
        return

def find_children(node):
    if not node.children:
        return [node]
    else:
        children = []
        for child in node.children:
            children += (find_children(child))
    return children

def builder(parent,threshold):    
    subdivision(parent,threshold)
    for i in range(len(parent.children)):
        builder(parent.children[i],threshold)

def subdivision(node,threshold):
    if len(node.points) <= threshold:
        return 
    dx,dy = node.width/2,node.height/2
    c1 = node.center[0]+dx/2,node.center[1]+dy/2
    p1 = point_selector(c1,dx,dy,node.points)
    n1 = Node(c1,dx,dy,p1)
    #print(f'Found {len(p1)} points top right')
    c2 = node.center[0]-dx/2,node.center[1]+dy/2
    p2 = point_selector(c2,dx,dy,node.points)
    n2 = Node(c2,dx,dy,p2)
    #print(f'Found {len(p2)} points top left')
    c3 = node.center[0]-dx/2,node.center[1]-dy/2
    p3 = point_selector(c3,dx,dy,node.points)
    n3 = Node(c3,dx,dy,p3)
    #print(f'Found {len(p3)} points bottom left')
    c4 = node.center[0]+dx/2,node.center[1]-dy/2
    p4 = point_selector(c4,dx,dy,node.points)
    n4 = Node(c4,dx,dy,p4)
    #print(f'Found {len(p4)} points bottom right')
    
    node.children = [n1,n2,n3,n4]
    
def point_selector(center,dx,dy,points):
    xmin,xmax = center[0]-dx/2,center[0]+dx/2
    ymin,ymax = center[1]-dy/2,center[1]+dy/2
    p = []
    #print(f'Box dim: {np.round(xmin,2)} < x < {np.round(xmax,2)}, {np.round(ymin,2)} < y < {np.round(ymax,2)} ')
    for i in points:
        if i.x >= xmin and i.y >= ymin and i.x < xmax and i.y < ymax:
            p.append(i)
    return p

if __name__ == '__main__':
    print('--- Exercise 7 ---')

    filename = 'colliding.hdf5'
    url = 'https://home.strw.leidenuniv.nl/~nobels/coursedata/'
    if not os.path.isfile(filename):
        print(f'File not found, downloading {filename}')
        os.system('wget '+url+filename)

    f = h5py.File(filename,'r')
    #print(list(f.keys()))
    a_group_key = list(f.keys())[1]
    data_type4 = f['PartType4']['Coordinates']
    data_type4 = data_type4[:,:2]

    t = Tree(12,data_type4)
    t.graph()
