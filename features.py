from data_reader import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from scipy import stats
from scipy.special import stdtr


def show_boxplot(feature):
    donated_true = []
    donated_false = []
    for i in range(len(y_all)):
        if y_all[i] == 1:
            donated_true.append(feature[i])
        else:
            donated_false.append(feature[i])
    #print donated_true
    #print donated_false
    data = [donated_false, donated_true]
    #labels = ['Donated = 0', 'Donated = 1']
    plt.boxplot(data)#, labels=labels)
    plt.show()

# load data without split
x_all, y_all = data_reader('data/9db113a1-cdbe-4b1c-98c2-11590f124dd8.csv', split=False)


# box plot existing features
feat_since_last_donation = x_all[:,1]
feat_number_donations = x_all[:,2]
feat_volume_donated = x_all[:,3]
feat_since_first_donation = x_all[:,4]

av_donation_volume = []
for i in range (len(y_all)):
    av_donation_volume.append(float(feat_volume_donated[i])/float(feat_number_donations[i]))
print 'average', av_donation_volume

feat_norm_since_first_donation = preprocessing.normalize(feat_since_last_donation)
print feat_norm_since_first_donation
feat_minmax_since_first_donation = preprocessing.minmax_scale(feat_since_last_donation)
print feat_minmax_since_first_donation
feat_npminmax_since_first_donation = (feat_norm_since_first_donation - feat_norm_since_first_donation.min()) / (feat_norm_since_first_donation.max() - feat_norm_since_first_donation.min())
print feat_npminmax_since_first_donation
x = feat_since_last_donation
minmax = [float(x_i - min(x)) / float(max(x) - min(x)) for x_i in x]
print minmax

#show_boxplot(feat_since_first_donation)
#show_boxplot(minmax)

t1, p1 = stats.ttest_ind(x, y_all, equal_var = False)
print float(p1)
print("ttest:            t = %g  p = %g" % (t1, float(p1)))
t2, p2 = stats.ttest_ind(minmax, y_all, equal_var = False)
print("ttest_minmax:            t = %g  p = %g" % (t2, float(p2)))

# eh?
# Compute the descriptive statistics of a and b.
abar = x.mean()
print type(x)
avar = x.var(ddof=1)
na = x.size
adof = na - 1

b = np.asarray(minmax)
bbar = b.mean()
bvar = b.var(ddof=1)
nb = b.size
bdof = nb - 1

# Use scipy.stats.ttest_ind_from_stats.
#t2, p2 = stats.ttest_ind_from_stats(abar, np.sqrt(avar), na, bbar, np.sqrt(bvar), nb, equal_var=False)
#print("ttest_ind_from_stats: t = %g  p = %g" % (t2, p2))

# Use the formulas directly.
tf = (abar - bbar) / np.sqrt(avar/na + bvar/nb)
dof = (avar/na + bvar/nb)**2 / (avar**2/(na**2*adof) + bvar**2/(nb**2*bdof))
pf = 2*stdtr(dof, -np.abs(tf))

print("formula:              t = %g  p = %g" % (tf, pf))

'''
# feature: donated only once. DO WE NEED IT?
# plotting.
num_donations = x_all[:,2]
print num_donations'''

'''
for i in range(len(y_all)):
    if num_donations[i] == 1:
        print num_donations[i], y_all[i], x_all[i]
        if y_all[i] == 1:
            print x_all[i]'''


'''


# what if we normalise it?

# ------ add feature 'donated once'

feat_donated_once = []
for i in range(len(y_all)):
    if num_donations[i] == 1:
        feat_donated_once.append(1)
    else:
        feat_donated_once.append(0)
print feat_donated_once

for i in range(len(y_all)):
    print feat_donated_once[i], num_donations[i]
#show_boxplot(feat_donated_once)
'''