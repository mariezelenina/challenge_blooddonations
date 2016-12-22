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

# Accuracy score
print trained_model.score(x_test,y_test)
# todo Log loss score


'''
# make submission file
x_submit = data_reader('data/5c9fa979-5a84-45d6-93b9-543d1a0efc41.csv', ycolumn=False)
y_submit_predicted = trained_model.predict(x_submit)

data_writer(x_submit,y_submit_predicted,'data/submissions/LogReg.csv')'''


