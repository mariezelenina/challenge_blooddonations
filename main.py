from data_reader import *
from sklearn import *

# load data
# default train/test split = 80/20
x_train, x_test, y_train, y_test = data_reader('data/9db113a1-cdbe-4b1c-98c2-11590f124dd8.csv')

# load model
model = linear_model.LogisticRegression()

# train model
trained_model = model.fit(x_train,y_train)

# do predictions
y_test_predicted = trained_model.predict(x_test)
print y_test_predicted

y_test_predicted_probs = trained_model.predict_proba(x_test)[:,1]
print y_test_predicted_probs

# Scores
print 'Score: accuracy ', trained_model.score(x_test,y_test)
print 'Score: log loss ', metrics.log_loss(y_test,y_test_predicted)
print 'Score: log loss for proba ', metrics.log_loss(y_test,y_test_predicted_probs)


#-------------------------------------------
# make submission file
x_submit = data_reader('data/5c9fa979-5a84-45d6-93b9-543d1a0efc41.csv', ycolumn=False)

# -> binary class prediction
y_submit_predicted = trained_model.predict(x_submit)
data_writer(x_submit,y_submit_predicted,'data/submissions/LogReg.csv')

# -> probs predictions
y_submit_predicted_probs = trained_model.predict_proba(x_submit)[:,1]
data_writer(x_submit,y_submit_predicted_probs,'data/submissions/LogReg_probs.csv')


