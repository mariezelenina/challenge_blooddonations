import numpy as np

class BloodDonations(object):

    def __init__(self, x_featdict_train, x_featdict_test, y_train, y_test):
        self.featdict_train = x_featdict_train
        self.featdict_test = x_featdict_test
        self.y_train = y_train
        self.y_test = y_test

    def add_feature(self, feature_name):
        #update x here
        if feature_name == 'A':
            print 'A'

    def remove_feature(self, feature_name):
        if feature_name in self.featdict_xtrain.keys():
            print 'remove it'
        else:
            print 'error'

    def create_inputready_x(self):
        # make input matrix out of feature dict
        x_train = np.array(self.featdict_train.values())
        x_test = np.array(self.featdict_test.values())

        self.x_train = x_train.transpose() # this is ready to feed to classifier
        self.x_test = x_test.transpose()   # this is ready to feed to classifier

        return self.x_train, self.x_test
