from __future__ import division
__author__ = 'Vladimir Iglovikov'

import math
import xgboost as xgb

def XGBT(X_train, y_train, X_test, y_test):
  '''

  :param X_train:
  :param y_train:
  :param X_test:
  :return:
  '''

  num_rounds = 10000

  params_new = {
  'objective': 'reg:linear',
  'eta': 0.001,
  'min_child_weight': 6,
  'subsample': 0.7,
  'colsample_bytree': 0.45,
  'scal_pos_weight': 1,
  'silent': 1,
  'max_depth': 7
  }

  offset = 5000

  xgtrain = xgb.DMatrix(X_train.values[offset:, :], label=y_train.values[offset:])
  xgval = xgb.DMatrix(X_train.values[:offset, :], label=y_train.values[:offset])
  xtest = xgb.DMatrix(X_test.values, label=y_test.values)


  watchlist = [(xgtrain, 'train'), (xgval, 'val')]
  model = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
  preds = model.predict(xgtest, ntree_limit=model.best_iteration)
  return preds
