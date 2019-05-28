#a2_6.py
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
from a2_1 import rng,box_muller


def train_perceptron (data_in,data_out,rng): 
    bias = np.zeros ((len(data_in),1)) # Initialise bias array.
    data_in = np.append(data_in,bias,axis=1) # Merge data_in and weights.
    weights = np.array([rng.rand_num(len(data_in[0])),rng.rand_num(len(data_in[0]))]).reshape(len(data_in[0]),2) # initialise random weights
    weighted_sum = np.dot(data_in,weights) # Compute weighted sum.
    output_indices = np.argmax(weighted_sum,axis=1) # Select maximum value.

    epoch = 0
    

    while epoch < 4000:
        wrongly_classified_indices = []
        for i in range(len(data_in)):
            if output_indices[i] != data_out[i]:
                wrongly_classified_indices.append(i)

        rand = rng.rand_num(len(wrongly_classified_indices)-1)*10
        k = wrongly_classified_indices[int(rand[0])]

        for j in range (2):
            if weighted_sum[k][j] > weighted_sum[k][int(data_out[k])]:
                weights[:,j] -= data_in[k]
            if j == int ( data_out[k] ):
                weights[:,j] += data_in[k]

        weighted_sum = np.dot(data_in,weights)
        output_indices = np.argmax(weighted_sum,axis = 1)

        epoch += 1

    sys.stdout.write("Iterations:{0}\n".format(epoch))

    #plt.show() # Toggle

    return weights

def test_perceptron(data_in,data_out,weights,hist=False):
    bias = np.zeros ((len(data_in ),1)) # Initialise bias array.
    data_in = np.append(data_in,bias,axis=1) # Merge data_in and weights.
    weighted_sum = np.dot(data_in, weights)
    output_indices = np.argmax(weighted_sum,axis = 1)
    correct_counter = 0
    for i in range (len(data_in)):
        if output_indices[i] == data_out[i]:
            correct_counter += 1
    print('Accuracy:{:03.1f}%\n'.format(correct_counter*100/len(data_out)))

    if hist: 
        short_true = len(data_out[data_out==0]) 
        short_predicted = len(output_indices[output_indices==0])
        long_true = len(data_out[data_out==1]) 
        long_predicted = len(output_indices[output_indices==1])
        bar_width = 0.35
        plt.bar(np.array([0,1])-bar_width/2,np.array([short_predicted,long_predicted]),bar_width,label='Predicted')
        plt.bar(np.array([0,1])+bar_width/2,np.array([short_true,long_true]),bar_width,label='True')
        plt.xticks((0,1))
        plt.xlabel('Labels')
        plt.ylabel('Number of GRBs')
        plt.legend()
        plt.title('Predicted labels for Gamma Ray Bursts')
        #plt.show()
        plt.savefig('./plots/6.png')
        plt.close()

if __name__ == '__main__':
    print('--- Exercise 6 ---')

    seed = 627310980
    print('Seed:',seed)
    rng = rng(seed)

    filename = 'GRBs.txt'
    url = 'https://home.strw.leidenuniv.nl/~nobels/coursedata/'
    if not os.path.isfile(filename):
        print(f'File not found, downloading {filename}')
        os.system('wget '+url+filename)

    data = np.genfromtxt(filename,skip_header=2,usecols = (2,3,4,5,6,7))
    data[data==-1.0] = 0
    names = np.genfromtxt(filename,skip_header=2,usecols=0,dtype=str)
    data = data[names!='XRF']
    labels = np.zeros(len(data))
    labels[data[:,1]>=10] += 1
    data = data[:,[0,2,3,4,5]]
    train_percent = 0.8
    train_in = data[:int(len(data)*train_percent)]
    train_out = labels[:int(len(labels)*train_percent)]
    test_in = data[int(len(data)*train_percent):]
    test_out = labels[int(len(labels)*train_percent):]

    for i in range(1):
        #sys.stdout.write("Run {0}\n".format(i+1))
        weights = train_perceptron(train_in, train_out,rng)
        print('Training set')
        test_perceptron(train_in,train_out,weights)
        print('Test set')
        test_perceptron(test_in,test_out,weights)
        print('Entire data set')
        test_perceptron(data,labels,weights,hist=True)
