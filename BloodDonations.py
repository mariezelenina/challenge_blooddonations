import numpy as np

class BloodDonations(object):

    def __init__(self, x_featdict_train, x_featdict_test, y_train, y_test):
        self.featdict_train = x_featdict_train
        self.featdict_test = x_featdict_test
        self.y_train = y_train
        self.y_test = y_test
        #todo train, test - make universal

    def add_feature(self, feature_name):
        if feature_name == 'DonationFrequency':
            # DonationFrequency = (number of donations)/(number of months as donor)
            # train
            feat_NumberOfDonations_train = self.featdict_train['feat_NumberOfDonations']
            number_months_train = self.featdict_train['feat_MonthsSinceFirstDonation'] - self.featdict_train['feat_MonthsSinceLastDonation']
            self.featdict_train['feat_DonationFrequency'] = number_months_train.astype(float)/feat_NumberOfDonations_train.astype(float)
            # test
            feat_NumberOfDonations_test = self.featdict_test['feat_NumberOfDonations']
            number_months_test = self.featdict_test['feat_MonthsSinceFirstDonation'] - self.featdict_test['feat_MonthsSinceLastDonation']
            self.featdict_test['feat_DonationFrequency'] = number_months_test.astype(float)/feat_NumberOfDonations_test.astype(float)
            return self.featdict_train, self.featdict_test
        elif feature_name == 'DonatedOnce':
            # train
            feat_NumberOfDonations_train = self.featdict_train['feat_NumberOfDonations']
            feat_donated_once = []
            for num_dons in feat_NumberOfDonations_train:
                if num_dons == 1:
                    feat_donated_once.append(1)
                else:
                    feat_donated_once.append(0)
            self.featdict_train['feat_DonatedOnce'] = np.array(feat_donated_once)
            # test
            feat_NumberOfDonations_test = self.featdict_test['feat_NumberOfDonations']
            feat_donated_once = []
            for num_dons in feat_NumberOfDonations_test:
                if num_dons == 1:
                    feat_donated_once.append(1)
                else:
                    feat_donated_once.append(0)
            self.featdict_test['feat_DonatedOnce'] = np.array(feat_donated_once)
            return self.featdict_train, self.featdict_test
        elif feature_name == 'DonatedOnceOrTwice':
            # train
            feat_NumberOfDonations_train = self.featdict_train['feat_NumberOfDonations']
            feat_donated_once = []
            for num_dons in feat_NumberOfDonations_train:
                if num_dons == 1 or num_dons == 2:
                    feat_donated_once.append(1)
                else:
                    feat_donated_once.append(0)
            self.featdict_train['feat_DonatedOnce'] = np.array(feat_donated_once)
            # test
            feat_NumberOfDonations_test = self.featdict_test['feat_NumberOfDonations']
            feat_donated_once = []
            for num_dons in feat_NumberOfDonations_test:
                if num_dons == 1 or num_dons == 2:
                    feat_donated_once.append(1)
                else:
                    feat_donated_once.append(0)
            self.featdict_test['feat_DonatedOnce'] = np.array(feat_donated_once)
            return self.featdict_train, self.featdict_test
        elif feature_name == 'SinceLastDonation-OneMonth':
            # train
            feat_MonthsSinceLastDonation_train = self.featdict_train['feat_MonthsSinceLastDonation']
            feature = []
            for months in feat_MonthsSinceLastDonation_train:
                if months == 1:
                    feature.append(1)
                else:
                    feature.append(0)
            self. featdict_train['feat_SinceLastDonation-OneMonth'] = np.array(feature)
            #test
            feat_MonthsSinceLastDonation_test = self.featdict_test['feat_MonthsSinceLastDonation']
            feature = []
            for months in feat_MonthsSinceLastDonation_test:
                if months == 1:
                    feature.append(1)
                else:
                    feature.append(0)
            self.featdict_test['feat_SinceLastDonation-OneMonth'] = np.array(feature)
            return self.featdict_train, self.featdict_test
        elif feature_name == 'SinceLastDonation-OneTwoMonths':
            # train
            feat_MonthsSinceLastDonation_train = self.featdict_train['feat_MonthsSinceLastDonation']
            feature = []
            for months in feat_MonthsSinceLastDonation_train:
                if months == 1 or months == 2:
                    feature.append(1)
                else:
                    feature.append(0)
            self.featdict_train['feat_SinceLastDonation-OneTwoMonth'] = np.array(feature)
            # test
            feat_MonthsSinceLastDonation_test = self.featdict_test['feat_MonthsSinceLastDonation']
            feature = []
            for months in feat_MonthsSinceLastDonation_test:
                if months == 1 or months == 2:
                    feature.append(1)
                else:
                    feature.append(0)
            self.featdict_test['feat_SinceLastDonation-OneTwoMonth'] = np.array(feature)
            return self.featdict_train, self.featdict_test

    def remove_feature(self, feature_name):
        if feature_name in self.featdict_xtrain.keys():
            del self.featdict_xtrain[feature_name]
            del self.featdict_xtest[feature_name]
        else:
            print 'error'
        return self.featdict_train, self.featdict_test

    def create_inputready_x(self):
        # make input matrix out of feature dict
        x_train = np.array(self.featdict_train.values())
        x_test = np.array(self.featdict_test.values())

        self.x_train = x_train.transpose() # this is ready to feed to classifier
        self.x_test = x_test.transpose()   # this is ready to feed to classifier

        return self.x_train, self.x_test
