from __future__ import division
from sklearn.cross_validation import ShuffleSplit
import pandas as pd
import math
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import xgboost as xgb
from gini_normalized import normalized_gini

'''
In this script I will try to find weights for averaging using cross validation
'''

print 'read joined'
joined = pd.read_csv('../data/joined.csv')

print 'splitting into train and test'

train = joined[joined['Hazard'] != -1]
test = joined[joined['Hazard'] == -1]

y = train['Hazard']
X = train.drop(['Hazard', 'Id', 'T2_V10', 'T2_V7', 'T1_V13', 'T1_V10'], 1)
X_test = test.drop(['Hazard', 'Id', 'T2_V10', 'T2_V7', 'T1_V13', 'T1_V10'], 1)

print 'applying log(1+x) to columns'

for column in X.columns:
  X[column] = X[column].apply(lambda x: math.log(1 + x))
  X_test[column] = X_test[column].apply(lambda x: math.log(1 + x))

test_size = 0.2
random_state = 42
offset = 5000
num_rounds = 1000000

params = {
  # 'objective': 'reg:linear',
  'objective': 'count:poisson',
  'eta': 0.005,
  'min_child_weight': 2,
  'subsample': 1,
  'colsample_bytree': 0.4,
  'scal_pos_weight': 1,
  'silent': 1,
  'max_depth': 7
}
params_new = list(params.items())

n_iter = 10

rs = ShuffleSplit(len(y), n_iter=n_iter, test_size=test_size, random_state=random_state)


score_00 = []
score_01 = []
score_02 = []
score_03 = []
score_04 = []
score_05 = []
score_06 = []
score_07 = []
score_08 = []
score_09 = []
score_1 = []

for train_index, test_index in rs:
  X_train = X.values[train_index]
  X_test = X.values[test_index]
  y_train = y.values[train_index]
  y_test = y.values[test_index]

  #fit RF
  n_estimators = 1000
  min_samples_split = 2
  max_features = 0.3
  max_depth = 15
  min_samples_leaf = 1

  clf1 = RandomForestRegressor(n_estimators=n_estimators,
                                          min_samples_split=min_samples_split,
                                          max_features=max_features,
                                          max_depth=max_depth,
                                          min_samples_leaf=min_samples_leaf,
                                          n_jobs=-1,
                                          random_state=random_state)

  clf1.fit(X_train, y_train)

  preds1 = clf1.predict(X_test)

  clf2 = RandomForestRegressor(n_estimators=n_estimators,
                              min_samples_split=min_samples_split,
                              max_features=max_features,
                              max_depth=max_depth,
                              min_samples_leaf=min_samples_leaf,
                              n_jobs=-1,
                              random_state=random_state)

  clf2.fit(X_train, np.log(y_train))

  preds2 = clf2.predict(X_test)

  preds_RF = 0.5 * preds1 + 0.5 * np.exp(preds2)

  #Let's try to do predictions using xgbt
  xgtest = xgb.DMatrix(X_test)

  xgtrain = xgb.DMatrix(X_train[offset:, :], label=y_train[offset:])
  xgval = xgb.DMatrix(X_train[:offset, :], label=y_train[:offset])

  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

  model = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)

  preds1 = model.predict(xgtest, ntree_limit=model.best_iteration)

  X_train = X_train[::-1, :]
  labels = np.log(y_train[::-1])

  xgtrain = xgb.DMatrix(X_train[offset:, :], label=labels[offset:])
  xgval = xgb.DMatrix(X_train[:offset, :], label=labels[:offset])

  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

  model = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)

  preds2 = model.predict(xgtest, ntree_limit=model.best_iteration)

  preds_xgbt = 0.5 * preds1 + 0.5 * np.exp(preds2)


  alpha = 0
  prediction = preds_xgbt
  tp = normalized_gini(y_test, prediction)
  score_00 += [normalized_gini(y_test, prediction)]
  alpha = 0.1
  prediction = alpha * preds_RF + (1 - alpha) * preds_xgbt
  score_01 += [normalized_gini(y_test, prediction)]

  alpha = 0.2
  prediction = alpha * preds_RF + (1 - alpha) * preds_xgbt
  score_02 += [normalized_gini(y_test, prediction)]

  alpha = 0.3
  prediction = alpha * preds_RF + (1 - alpha) * preds_xgbt
  score_03 += [normalized_gini(y_test, prediction)]

  alpha = 0.4
  prediction = alpha * preds_RF + (1 - alpha) * preds_xgbt
  score_04 += [normalized_gini(y_test, prediction)]

  alpha = 0.5
  prediction = alpha * preds_RF + (1 - alpha) * preds_xgbt
  score_05 += [normalized_gini(y_test, prediction)]

  alpha = 0.6
  prediction = alpha * preds_RF + (1 - alpha) * preds_xgbt
  score_06 += [normalized_gini(y_test, prediction)]

  alpha = 0.7
  prediction = alpha * preds_RF + (1 - alpha) * preds_xgbt
  score_07 += [normalized_gini(y_test, prediction)]

  alpha = 0.8
  prediction = alpha * preds_RF + (1 - alpha) * preds_xgbt
  score_08 += [normalized_gini(y_test, prediction)]

  alpha = 0.9
  prediction = alpha * preds_RF + (1 - alpha) * preds_xgbt
  score_09 += [normalized_gini(y_test, prediction)]

  alpha = 1
  prediction = preds_RF
  score_1 += [normalized_gini(y_test, prediction)]


print '0'
print np.mean(score_00), np.std(score_00)
print '0.1'
print np.mean(score_01), np.std(score_01)
print '0.2'
print np.mean(score_02), np.std(score_02)
print '0.3'
print np.mean(score_03), np.std(score_03)
print '0.4'
print np.mean(score_04), np.std(score_04)
print '0.5'
print np.mean(score_05), np.std(score_05)
print '0.6'
print np.mean(score_06), np.std(score_06)
print '0.7'
print np.mean(score_07), np.std(score_07)
print '0.8'
print np.mean(score_08), np.std(score_08)
print '0.9'
print np.mean(score_09), np.std(score_09)
print '1'
print np.mean(score_1), np.std(score_1)




