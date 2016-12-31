from sklearn import *
from collections import Counter
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_reader import *



dict = {'feat_a': ['a1', 'a2', 'a3'],
        'feat_b': ['b1', 'b2', 'b3'],
        'feat_c': ['c1', 'c2', 'c3'],
        'feat_d': ['d1', 'd2', 'd3']}

print dict

print np.array(dict.values())
a = np.array(dict.values())

aa=a.transpose()
print aa


'''
def data_reader(path, ycolumn = True, split = True, trainsize = 0.8):
    # read the file into numpy_array. Split features and output values.
    datafile = open(path, 'r')
    df = pd.read_csv(datafile, header=0, delimiter=",")
    numpy_array = df.as_matrix()
    print numpy_array[0]

    # if we are loading submission test data, we don't have any y
    if not ycolumn:
        return numpy_array
    else:
        dimension = numpy_array.shape[1]-1

        y_all = numpy_array[:,-1]
        x_all = np.delete(numpy_array,dimension,1)

        if split:
            x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size = 1-trainsize, random_state = 0)
            #todo - randomness of split
            return x_train, x_test, y_train, y_test
        else:
            return x_all, y_all

x_train, x_test, y_train, y_test = data_reader('data/9db113a1-cdbe-4b1c-98c2-11590f124dd8.csv')

'''