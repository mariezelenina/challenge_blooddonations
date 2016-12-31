from sklearn import *
from collections import Counter
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_reader import *

# THIS FILE IS USED ONCE AND WON'T BE CHANGED AFTERWARDS

#baseline score
x_train, x_test, y_train, y_test = data_reader('data/9db113a1-cdbe-4b1c-98c2-11590f124dd8.csv')

y_train.tolist
y_test.tolist
print Counter(y_train)

y_test_predicted = [0] * len(y_test)

print "BASELINE:"
print 'Acc:', accuracy_score(y_test, y_test_predicted)
print 'Log Loss:', metrics.log_loss(y_test,y_test_predicted)
