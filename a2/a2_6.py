#a2_6.py
import numpy as np
import sys
import matplotlib.pyplot as plt


def train_perceptron (data_in,data_out): 
    bias = np.zeros ((len(data_in),1)) # Initialise bias array.
    data_in = np.append(data_in,bias,axis=1) # Merge data_in and weights.
    weights = np.random.rand (len(data_in[0]),2) # Initialise weights array.randomly.
    weighted_sum = np.dot(data_in,weights) # Compute weighted sum.
    output_indices = np.argmax(weighted_sum,axis=1) # Select maximum value.

    epoch = 0

    while epoch < 5000:#np.array_equal(output_indices,data_out) == False:
        wrongly_classified_indices = []
        for i in range(len(data_in)):
            if output_indices[i] != data_out[i]:
                wrongly_classified_indices.append(i)

        if epoch % 10 == 0: 
            plt.plot(epoch,(len(data_in)-len(wrongly_classified_indices)) * 100/float(len(data_in)),'.')

        k = wrongly_classified_indices[np.random.randint(0,len(wrongly_classified_indices)-1)]

        for j in range (2):
            if weighted_sum[k][j] > weighted_sum[k][int(data_out[k])]:
                weights[:,j] -= data_in[k]
            if j == int ( data_out[k] ):
                weights[:,j] += data_in[k]

        weighted_sum = np.dot(data_in,weights)
        output_indices = np.argmax(weighted_sum,axis = 1)

        epoch += 1

    sys.stdout.write("Iterations:{0}\n".format(epoch))

    plt.show() # Toggle

    return weights

def test_perceptron(data_in,data_out,weights):
    bias = np.zeros ((len(data_in ),1)) # Initialise bias array.
    data_in = np.append(data_in,bias,axis=1) # Merge data_in and weights.
    weighted_sum = np.dot(data_in, weights)
    output_indices = np.argmax(weighted_sum,axis = 1)
    correct_counter = 0
    for i in range (len(data_in)):
        if output_indices[i] == data_out[i]:
            correct_counter += 1
    sys.stdout.write( "Accuracy: {:03.1f}%\n".format(correct_counter*100/len(data_out)))



if __name__ == '__main__':
    print('--- Exercise 5 ---')

    filename = 'GRBs.txt'
    url = 'https://home.strw.leidenuniv.nl/~nobels/coursedata/'
    if not os.path.isfile(filename):
        print(f'File not found, downloading {filename}')
        os.system('wget '+url+filename)

    random_num = np.genfromtxt(filename,delimiter=' ',skip_header=2)

