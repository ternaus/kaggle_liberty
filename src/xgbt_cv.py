from __future__ import division
__author__ = 'Vladimir Iglovikov'

'''
Here I will try to use xgb.
'''
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import ShuffleSplit
import numpy as np
from gini_normalized import normalized_gini
from preprocessing.to_labels import to_labels

train = pd.read_csv('../data/train_new.csv')
hold = pd.read_csv('../data/hold_new.csv')

train, test = to_labels(train, hold)

y = train['Hazard']
# X = train.drop(['Hazard', 'Id'], 1)
# X = train.drop(['Hazard', 'Id', 'T2_V10', 'T2_V7', 'T1_V13', 'T1_V10', 'tp_59', 'tp_84', 'global_mean', 'global_median', 'global_std'], 1)
# X_test = test.drop(['Hazard', 'Id', 'T2_V10', 'T2_V7', 'T1_V13', 'T1_V10', 'tp_59', 'tp_84', 'global_mean', 'global_median', 'global_std'], 1)
X = train.drop(['Hazard', 'Id', 'T2_V10', 'T2_V7', 'T1_V13', 'T1_V10'], 1)
X_test = test.drop(['Hazard', 'Id', 'T2_V10', 'T2_V7', 'T1_V13', 'T1_V10'], 1)

#


params = {
  'objective': 'reg:linear',
  # 'eta': 0.005,
  # 'min_child_weight': 6,
  # 'subsample': 0.7,
  # 'colsabsample_bytree': 0.7,
  # 'scal_pos_weight': 1,
  'silent': 1,
  # 'max_depth': 9
}

num_rounds = 10000
random_state = 42
offset = 5000

ind = 1
if ind == 1:
  n_iter = 10
  rs = ShuffleSplit(len(y), n_iter=n_iter, test_size=0.1, random_state=random_state)

  result = []

  for min_child_weight in [3]:
    for eta in [0.001]:
      for colsample_bytree in [0.5]:
        for max_depth in [7]:
          for subsample in [0.7]:
            for gamma in [1]:
              params['min_child_weight'] = min_child_weight
              params['eta'] = eta
              params['colsample_bytree'] = colsample_bytree
              params['max_depth'] = max_depth
              params['subsample'] = subsample
              params['gamma'] = gamma
              
              params_new = list(params.items())
              score = []
              for train_index, test_index in rs:

                X_train = X.values[train_index]
                X_test = X.values[test_index]
                y_train = y.values[train_index]
                y_test = y.values[test_index]

                xgtest = xgb.DMatrix(X_test)

                xgtrain = xgb.DMatrix(X_train[offset:, :], label=y_train[offset:])
                xgval = xgb.DMatrix(X_train[:offset, :], label=y_train[:offset])

                watchlist = [(xgtrain, 'train'), (xgval, 'val')]

                model = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)

                preds = model.predict(xgtest, ntree_limit=model.best_iteration)
                # preds = model.predict(xgval, ntree_limit=model.best_iteration)

                tp = normalized_gini(y_test, preds)
                # tp = normalized_gini(y_train[:offset], preds)
                score += [tp]
                print tp

              result += [(np.mean(score), np.std(score), min_child_weight, eta, colsample_bytree, max_depth, subsample, gamma, n_iter)]

  result.sort()
  print result

elif ind == 2:
  X_test.fillna(-1, inplace=True)
  xgtrain = xgb.DMatrix(X.values[offset:, :], label=y.values[offset:])
  xgval = xgb.DMatrix(X.values[:offset, :], label=y.values[:offset])
  xgtest = xgb.DMatrix(X_test.values)
  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

  params = {
  'objective': 'reg:linear',
  'eta': 0.005,
  'min_child_weight': 3,
  'subsample': 0.7,
  'colsample_bytree': 0.5,
  # 'scal_pos_weight': 1,
  'silent': 1,
  'max_depth': 7,
  'gamma': 1
  }    
  params_new = list(params.items())
  model = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
  prediction = model.predict(xgtest, ntree_limit=model.best_iteration)
  
  submission = pd.DataFrame()
  submission['Id'] = test['Id']
  submission['Hazard'] = prediction
  submission.to_csv("xgbt/xgbt_6.csv", index=False)
