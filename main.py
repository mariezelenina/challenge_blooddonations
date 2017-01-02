from data_reader import *
from sklearn import *
from BloodDonations import *

nameOfExperiment = 'LogReg_Plus-All'
featuresToAdd = ['DonationFrequency',
                 'DonatedOnce',
                 'DonatedOnceOrTwice',
                 'SinceLastDonation-OneMonth',
                 'SinceLastDonation-OneTwoMonth']

# load data
path = 'data/9db113a1-cdbe-4b1c-98c2-11590f124dd8.csv'
x_ids_train, x_ids_test, feat_dict_xtrain, feat_dict_xtest, y_train, y_test = data_reader(path)

# put data into class, add/remove features as desired
MyData_train = BloodDonations(x_ids_train, feat_dict_xtrain,y_train)
MyData_test = BloodDonations(x_ids_test, feat_dict_xtest,y_test)

for featurename in featuresToAdd:
    MyData_train.add_feature(featurename)
    MyData_test.add_feature(featurename)
MyData_train.create_inputready_x()
MyData_test.create_inputready_x()
'''MyData.add_feature('DonationFrequency')
MyData.add_feature('DonatedOnce')
MyData.add_feature('DonatedOnceOrTwice')
MyData.add_feature('SinceLastDonation-OneMonth')
MyData.add_feature('SinceLastDonation-OneTwoMonth')
'''

# do classification:
# --> load model
model = linear_model.LogisticRegression()

# --> train model
trained_model = model.fit(MyData_train.x, MyData_train.y)

# --> do predictions
y_test_predicted = trained_model.predict(MyData_test.x)
# print y_test_predicted

y_test_predicted_probs = trained_model.predict_proba(MyData_test.x)[:,1]
#print y_test_predicted_probs

print 'Y_correct', MyData_test.y
print 'Y_predic', y_test_predicted

# --> Scores
print 'Score: accuracy ', trained_model.score(MyData_test.x,MyData_test.y)
print 'Score: log loss ', metrics.log_loss(MyData_test.y,y_test_predicted)
print 'Score: log loss for proba', metrics.log_loss(MyData_test.y,y_test_predicted_probs)


#-------------------------------------------
# make submission file
x_ids, x_submit = data_reader('data/5c9fa979-5a84-45d6-93b9-543d1a0efc41.csv', ycolumn=False)

MyData_predict = BloodDonations(x_ids, x_submit,[])
for featurename in featuresToAdd:
    MyData_predict.add_feature(featurename)
MyData_predict.create_inputready_x()

# -> binary class prediction
y_submit_predicted = trained_model.predict(MyData_predict.x)
data_writer(MyData_predict.x_ids,y_submit_predicted,'data/submissions/'+nameOfExperiment+'.csv')

# -> probs predictions
y_submit_predicted_probs = trained_model.predict_proba(MyData_predict.x)[:,1]
data_writer(MyData_predict.x_ids,y_submit_predicted_probs,'data/submissions/'+nameOfExperiment+'_probs.csv')


