from __future__ import division
try:
  from src import Predict
except:
  pass

import sys
sys.path += ['Predict']
from sklearn.linear_model import LinearRegression

try:
  from src.Predict import XGB
except:
  pass

try:
  from src.Predict import NN
except:
  pass

import XGB
import NN

__author__ = 'Vladimir Iglovikov'

'''
This script will use cross validation to estimate weights for merging of the different models.
(I will try with hold out set first)
'''

import pandas as pd
import os
from sklearn.cross_validation import train_test_split

random_state = 42

'''
[1] read train data and cut hold out set out of it
'''

joined = pd.read_csv(os.path.join('..', 'data', 'joined.csv'))

train = joined[joined['Hazard'] != -1]

y = train['Hazard']
X = train.drop(['Id', 'Hazard'], 1)

#cut holdout set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

'''
[2] Do XGB and NN simulation on the train set and create prediction on the hold out set.
'''

nn_prediction = NN.NN(X_train, y_train, X_test, y_test)
xgb_prediction = XGB.XGB(X_train, y_train, X_test, y_test)


'''
[3] Merge previous predictions into dataset and do linear regression on it
'''

result_train = pd.DataFrame()
result_train['nn'] = nn_prediction
result_train['xgb'] = xgb_prediction

clf = LinearRegression(n_jobs=-1)
clf.fit(result_train, y_test)

print clf.intercept_
print clf.coef_
