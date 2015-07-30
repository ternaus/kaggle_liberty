from __future__ import division
import numpy as np

try:
  from src import Predict
except:
  pass

import sys
sys.path += ['Predict']
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation

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
'''

import pandas as pd
import os
from sklearn.cross_validation import train_test_split

random_state = 42

xgb_test = pd.read_csv('predictions1/1438258857.28.csv')
nn_test = pd.read_csv('predictions/1438258912.82.csv')

'''
[1] read train data and cut hold out set out of it
'''

joined = pd.read_csv(os.path.join('..', 'data', 'joined.csv'))

train = joined[joined['Hazard'] != -1]

y = train['Hazard']
X = train.drop(['Id', 'Hazard'], 1)

kf = cross_validation.KFold(len(y), n_folds=5, random_state=random_state)

result_nn = pd.DataFrame()
result_xgb = pd.DataFrame()

ind = 0
for train_index, test_index in kf:
  ind += 1
  X_train = X[train_index]
  y_train = y[train_index]

  X_test = X[test_index]
  y_test = y[test_index]

  '''
  [2] Do XGB and NN simulation on the train set and create prediction on the hold out set.
  '''

  nn_prediction = NN.NN(X_train, y_train, X_test, y_test, uselog=True)
  xgb_prediction = XGB.XGB(X_train, y_train, X_test, y_test, uselog=True)


  # print np.reshape(nn_prediction, len(nn_prediction))
  '''
  [3] Merge previous predictions into dataset and do linear regression on it
  '''

  result_train = pd.DataFrame()

  result_nn[ind] = nn_prediction
  result_xgb[ind] = xgb_prediction

  result_train['nn'] = result_nn.mean(axis=1)
  result_train['xgb'] = result_xgb.mean(axis=1)

  # result_train['nn'] = nn_prediction
  # result_train['xgb'] = xgb_prediction

clf = LinearRegression(n_jobs=-1)
clf.fit(result_train, y_test)

print 'intercept = ', clf.intercept_
print 'coef_ = ', clf.coef_

result_test = pd.DataFrame()

result_test['nn'] = nn_test['Hazard']
result_test['xgb'] = xgb_test['Hazard']

final_prediction = clf.predict(result_test)

submission = pd.DataFrame()
submission['Id'] = nn_test['Id']
submission['Hazard'] = final_prediction

submission.to_csv('linear/nn_xgb_cv5.csv', index=False)