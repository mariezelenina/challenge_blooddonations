import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def make_dict(raw_x):
    feat_dict_x = {'feat_MonthsSinceLastDonation': raw_x[:, 0],
                    'feat_NumberOfDonations': raw_x[:, 1],
                    'feat_TotalVolumeDonated': raw_x[:, 2],
                    'feat_MonthsSinceFirstDonation': raw_x[:, 3]}
    return feat_dict_x


def data_reader(path, ycolumn = True, split = True, trainsize = 0.8):
    # read the file into numpy_array. Split features and output values.
    datafile = open(path, 'r')
    df = pd.read_csv(datafile, header=0, delimiter=",")
    numpy_array = df.as_matrix()

    # if we are loading submission test data, we don't have any y
    if not ycolumn:
        x_ids = numpy_array[:,0]
        x_all = np.delete(numpy_array, 0, 1)
        x_all_featdict = make_dict(x_all)
        return x_ids, x_all_featdict
    else:
        dimension = numpy_array.shape[1]-1

        y_all = numpy_array[:,-1]
        x_ids = numpy_array[:, 0]
        x_all = np.delete(numpy_array,dimension,1) #delete the y column

        if split:
            x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size = 1-trainsize, random_state = 0)
            x_ids_train = x_train[:, 0]
            x_train = np.delete(x_train,0,1)
            x_ids_test = x_test[:, 0]
            x_test = np.delete(x_test,0,1)
            x_train_featdict = make_dict(x_train)
            x_test_featdict = make_dict(x_test)
            return x_ids_train, x_ids_test, x_train_featdict, x_test_featdict, y_train, y_test
        else:
            x_all = np.delete(x_all, 0, 1)  # delete the id column
            x_all_featdict = make_dict(x_all)
            return x_ids, x_all_featdict, y_all

def data_writer(x_ids, data_y, filename):
    # creates a csv file ready for submission
    output = np.c_[x_ids,data_y]
    np.savetxt(filename,output,fmt='%10.5f',header=",Made Donation in March 2007",delimiter=',')