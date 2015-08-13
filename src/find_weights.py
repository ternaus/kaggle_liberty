from __future__ import division
import numpy as np

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
import Ridge
import RF

__author__ = 'Vladimir Iglovikov'

'''
This script will use cross validation to estimate weights for merging of the different models.
(I will try with hold out set first)
'''

import pandas as pd
import os
from sklearn.cross_validation import train_test_split

random_state = 42

# xgb_test = pd.read_csv('predictions1/1438258857.28.csv')
# nn_test = pd.read_csv('predictions/1438258912.82.csv')


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
rf_prediction = RF.RF(X_train, y_train, X_test, y_test)
ridge_prediction = Ridge.Ridge(X_train, y_train, X_test, y_test)


# print np.reshape(nn_prediction, len(nn_prediction))
'''
[3] Merge previous predictions into dataset and do linear regression on it
'''

result_train = pd.DataFrame()

result_train['nn'] = nn_prediction
result_train['xgb'] = xgb_prediction
result_train['rf'] = rf_prediction
result_train['ridge'] = ridge_prediction

clf = LinearRegression(n_jobs=-1)
clf.fit(result_train, y_test)

print 'intercept = ', clf.intercept_
print 'coef_ = ', clf.coef_
#
# result_test = pd.DataFrame()
#
# result_test['nn'] = nn_test['Hazard']
# result_test['xgb'] = xgb_test['Hazard']
#
#
# final_prediction = clf.predict(result_test)
#
# submission = pd.DataFrame()
# submission['Id'] = nn_test['Id']
# submission['Hazard'] = final_prediction
#
# submission.to_csv('linear/nn_xgb.csv', index=False)