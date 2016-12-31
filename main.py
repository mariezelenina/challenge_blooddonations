from data_reader import *
from sklearn import *
from BloodDonations import *

# load data
path = 'data/9db113a1-cdbe-4b1c-98c2-11590f124dd8.csv'
feat_dict_xtrain, feat_dict_xtest, y_train, y_test = data_reader(path)

# put data into class, add/remove features as desired
MyData = BloodDonations(feat_dict_xtrain,feat_dict_xtest,y_train,y_test)
MyData.add_feature('DonationFrequency')
MyData.add_feature('DonatedOnce')
MyData.add_feature('DonatedOnceOrTwice')
MyData.add_feature('SinceLastDonation-OneMonth')
MyData.add_feature('SinceLastDonation-OneTwoMonth')
MyData.create_inputready_x()
#print MyData.create_inputready_x()[0]

# do classification:
# --> load model
model = linear_model.LogisticRegression()

# --> train model
trained_model = model.fit(MyData.x_train, MyData.y_train)

# --> do predictions
y_test_predicted = trained_model.predict(MyData.x_test)
# print y_test_predicted

y_test_predicted_probs = trained_model.predict_proba(MyData.x_test)[:,1]
#print y_test_predicted_probs

print 'Y_correct', MyData.y_test
print 'Y_ predic', y_test_predicted

# --> Scores
print 'Score: accuracy ', trained_model.score(MyData.x_test,MyData.y_test)
print 'Score: log loss ', metrics.log_loss(MyData.y_test,y_test_predicted)
print 'Score: log loss for proba', metrics.log_loss(MyData.y_test,y_test_predicted_probs)


#-------------------------------------------
# make submission file
x_submit = data_reader('data/5c9fa979-5a84-45d6-93b9-543d1a0efc41.csv', ycolumn=False)

# -> binary class prediction
y_submit_predicted = trained_model.predict(x_submit)
data_writer(x_submit,y_submit_predicted,'data/submissions/LogReg_plus_all.csv')

# -> probs predictions
y_submit_predicted_probs = trained_model.predict_proba(x_submit)[:,1]
data_writer(x_submit,y_submit_predicted_probs,'data/submissions/LogReg_plus_all_probs.csv')


