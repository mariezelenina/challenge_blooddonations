import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def data_reader(path, ycolumn = True, split = True, trainsize = 0.8):
    # read the file into numpy_array. Split features and output values.
    datafile = open(path, 'r')
    df = pd.read_csv(datafile, header=0, delimiter=",")
    numpy_array = df.as_matrix()

    # if we are loading submission test data, we don't have any y
    if not ycolumn:
        return numpy_array
    else:
        dimension = numpy_array.shape[1]-1

        y_all = numpy_array[:,-1]
        x_all = np.delete(numpy_array,dimension,1)

        if split:
            x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size = 1-trainsize, random_state = 42)
            #todo - randomness of split
            return x_train, x_test, y_train, y_test
        else:
            return x_all, y_all


def data_writer(data_x, data_y, filename):
    # creates a csv file ready for submission
    output_x = data_x[:,0]
    output = np.c_[output_x,data_y]
    np.savetxt(filename,output,fmt='%10.5f',header=",Made Donation in March 2007",delimiter=',')