import numpy as np 
import gzip
import cPickle
import pandas as pd

def load_data():
    train_data = pd.read_csv("train.csv")
    data = np.array(train_data)
    training_result = data[0:42000, 0:1]
    training_inputs = data[0:42000, 1:]
    train_inputs = [np.reshape(x, (784, 1)) for x in training_inputs]
	#-> In above, x represents each row. In training_inputs,
    #   training_inputs[0].shape gives (784,) but we want
    #   (784,1). Hence we reshape the array.
    train_results = [vectorized_result(y) for y in training_result]
    training_data = zip(train_inputs, train_results)
	#-> Zip is used to return a list of tuples where in zip(x,y),
    #   each element in list is a tuple containing each element in x and y pairwise
    #valid_data = pd.read_csv("test.csv")
    #data = np.array(valid_data)
    #test_inputs	 = data[0:28000, 0:]

    return (training_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e



#def main():
#	part1()


#if __name__ == '__main__':
#		main()