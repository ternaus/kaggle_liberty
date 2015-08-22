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
test = pd.read_csv('../data/test.csv')

par = (train, hold, test)
train, hold, test = to_labels(par)

y = train['Hazard']
# X = train.drop(['Hazard', 'Id'], 1)
# X = train.drop(['Hazard', 'Id', 'T2_V10', 'T2_V7', 'T1_V13', 'T1_V10', 'tp_59', 'tp_84', 'global_mean', 'global_median', 'global_std'], 1)
# X_test = test.drop(['Hazard', 'Id', 'T2_V10', 'T2_V7', 'T1_V13', 'T1_V10', 'tp_59', 'tp_84', 'global_mean', 'global_median', 'global_std'], 1)
X = train.drop(['Hazard', 'Id', 'T2_V10', 'T2_V7', 'T1_V13', 'T1_V10'], 1)
X_hold = hold.drop(['Hazard', 'Id', 'T2_V10', 'T2_V7', 'T1_V13', 'T1_V10'], 1)
X_test = test.drop(['Id', 'T2_V10', 'T2_V7', 'T1_V13', 'T1_V10'], 1)

#


params = {
  # 'objective': 'reg:linear',
  'objective': 'count:poisson',
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
  result_truncated_up = []
  result_truncated_down = []
  result_truncated_both = []
  result_truncated_both_round = []
  result_truncated_both_int = []


  for min_child_weight in [3]:
    for eta in [0.01]:
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
              score_truncated_up = []
              score_truncated_down = []
              score_truncated_both = []
              score_truncated_both_round = []
              score_truncated_both_int = []

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

                preds1 = model.predict(xgtest, ntree_limit=model.best_iteration)

                X_train = X_train[::-1, :]
                labels = y_train[::-1]

                xgtrain = xgb.DMatrix(X_train[offset:, :], label=labels[offset:])
                xgval = xgb.DMatrix(X_train[:offset, :], label=labels[:offset])

                watchlist = [(xgtrain, 'train'), (xgval, 'val')]

                model = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)

                preds2 = model.predict(xgtest, ntree_limit=model.best_iteration)

                # preds = model.predict(xgval, ntree_limit=model.best_iteration)

                preds = 0.5 * preds1 + 0.5 * preds2

                tp = normalized_gini(y_test, preds)
                tp_up = normalized_gini(y_test, map(lambda x: min(69, x), preds))
                tp_down = normalized_gini(y_test, map(lambda x: max(1, x), preds))
                tp_both = normalized_gini(y_test, map(lambda x: min(69, max(1, x)), preds))
                tp_both_round = normalized_gini(y_test, map(lambda x: round(min(69, max(1, x))), preds))
                tp_both_int = normalized_gini(y_test, map(lambda x: int(min(69, max(1, x))), preds))

                # tp = normalized_gini(y_train[:offset], preds)
                score += [tp]
                score_truncated_up += [tp_up]
                score_truncated_down += [tp_down]
                score_truncated_both += [tp_both]
                score_truncated_both_int += [tp_both_int]
                score_truncated_both_round += [tp_both_round]
                print tp

              result += [(np.mean(score), np.std(score), min_child_weight, eta, colsample_bytree, max_depth, subsample, gamma, n_iter)]
              result_truncated_up += [(np.mean(score_truncated_up), np.std(score_truncated_up), min_child_weight, eta, colsample_bytree, max_depth, subsample, gamma, n_iter)]
              result_truncated_down += [(np.mean(score_truncated_down), np.std(score_truncated_down), min_child_weight, eta, colsample_bytree, max_depth, subsample, gamma, n_iter)]
              result_truncated_both += [(np.mean(score_truncated_both), np.std(score_truncated_both), min_child_weight, eta, colsample_bytree, max_depth, subsample, gamma, n_iter)]
              result_truncated_both_int += [(np.mean(score_truncated_both_int), np.std(score_truncated_both_int), min_child_weight, eta, colsample_bytree, max_depth, subsample, gamma, n_iter)]
              result_truncated_both_round += [(np.mean(score_truncated_both_round), np.std(score_truncated_both_round), min_child_weight, eta, colsample_bytree, max_depth, subsample, gamma, n_iter)]

  result.sort()
  result_truncated_up.sort()
  result_truncated_down.sort()
  result_truncated_both.sort()
  result_truncated_both_int.sort()
  result_truncated_both_round.sort()

  print
  print 'result'
  print result

  print
  print 'result_truncated_up'
  print result_truncated_up

  print
  print 'result_truncated_down'
  print result_truncated_down

  print
  print 'result truncated both'
  print result_truncated_both

  print
  print 'result truncated both int'
  print result_truncated_both_int
  print
  print 'result truncated both round'
  print result_truncated_both_round

elif ind == 2:
  xgtrain = xgb.DMatrix(X.values[offset:, :], label=y.values[offset:])
  xgval = xgb.DMatrix(X.values[:offset, :], label=y.values[:offset])
  xghold = xgb.DMatrix(X_hold.values)
  xgtest = xgb.DMatrix(X_test.values)

  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

  params = {
  # 'objective': 'reg:linear',
    'objective': 'count:poisson',
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
  prediction_hold = model.predict(xghold, ntree_limit=model.best_iteration)
  
  submission = pd.DataFrame()
  submission['Id'] = hold['Id']
  submission['Hazard'] = prediction_hold
  submission.to_csv("preds_on_hold/xgbt.csv", index=False)

  prediction_test = model.predict(xgtest, ntree_limit=model.best_iteration)

  submission = pd.DataFrame()
  submission['Id'] = test['Id']
  submission['Hazard'] = prediction_test
  submission.to_csv("preds_on_test/xgbt.csv", index=False)


elif ind == 3:
  xgtrain = xgb.DMatrix(X.values[offset:, :], label=y.values[offset:])
  xgval = xgb.DMatrix(X.values[:offset, :], label=y.values[:offset])
  xghold = xgb.DMatrix(X_hold.values)
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
  prediction = model.predict(xghold, ntree_limit=model.best_iteration)
  print 'score on the hold = ', normalized_gini(hold['Hazard'], prediction)