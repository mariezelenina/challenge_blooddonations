import numpy as np

class BloodDonations(object):

    def __init__(self, x_ids, x_featdict, y_train):
        self.x_ids = x_ids
        self.x_featdict = x_featdict
        self.y = y_train
        self.x = self.create_inputready_x()

    def add_feature(self, feature_name):
        if feature_name == 'DonationFrequency':
            # DonationFrequency = (number of donations)/(number of months as donor)
            feat_NumberOfDonations_train = self.x_featdict['feat_NumberOfDonations']
            number_months_train = self.x_featdict['feat_MonthsSinceFirstDonation'] - self.x_featdict['feat_MonthsSinceLastDonation']
            self.x_featdict['feat_DonationFrequency'] = number_months_train.astype(float)/feat_NumberOfDonations_train.astype(float)
            return self.x_featdict

        elif feature_name == 'DonatedOnce':
            # train
            feat_NumberOfDonations_train = self.x_featdict['feat_NumberOfDonations']
            feat_donated_once = []
            for num_dons in feat_NumberOfDonations_train:
                if num_dons == 1:
                    feat_donated_once.append(1)
                else:
                    feat_donated_once.append(0)
            self.x_featdict['feat_DonatedOnce'] = np.array(feat_donated_once)
            return self.x_featdict

        elif feature_name == 'DonatedOnceOrTwice':
            # train
            feat_NumberOfDonations_train = self.x_featdict['feat_NumberOfDonations']
            feat_donated_once = []
            for num_dons in feat_NumberOfDonations_train:
                if num_dons == 1 or num_dons == 2:
                    feat_donated_once.append(1)
                else:
                    feat_donated_once.append(0)
            self.x_featdict['feat_DonatedOnce'] = np.array(feat_donated_once)
            return self.x_featdict

        elif feature_name == 'SinceLastDonation-OneMonth':
            # train
            feat_MonthsSinceLastDonation_train = self.x_featdict['feat_MonthsSinceLastDonation']
            feature = []
            for months in feat_MonthsSinceLastDonation_train:
                if months == 1:
                    feature.append(1)
                else:
                    feature.append(0)
            self.x_featdict['feat_SinceLastDonation-OneMonth'] = np.array(feature)
            return self.x_featdict
        elif feature_name == 'SinceLastDonation-OneTwoMonths':
            # train
            feat_MonthsSinceLastDonation_train = self.x_featdict['feat_MonthsSinceLastDonation']
            feature = []
            for months in feat_MonthsSinceLastDonation_train:
                if months == 1 or months == 2:
                    feature.append(1)
                else:
                    feature.append(0)
            self.x_featdict['feat_SinceLastDonation-OneTwoMonth'] = np.array(feature)
            return self.x_featdict
        else:
            print 'Don\'t know this feature name '
            #todo throw proper error

    def remove_feature(self, feature_name):
        if feature_name in self.x_featdict.keys():
            del self.x_featdict[feature_name]
        else:
            print 'error'
        return self.x_featdict

    def create_inputready_x(self):
        # make input matrix out of feature dict
        x = np.array(self.x_featdict.values())
        self.x = x.transpose() # this is ready to feed to classifier
        return self.x
