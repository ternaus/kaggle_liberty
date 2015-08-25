from __future__ import division
__author__ = 'Vladimir Iglovikov'

'''
3 days left. I do not have any bright ideas right now, so I will try to
find single model with best parameters and average with other predictors
'''
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import ShuffleSplit
import numpy as np
from gini_normalized import normalized_gini
from preprocessing.to_labels import to_labels
import math
from sklearn.cross_validation import ShuffleSplit

print 'read joined'
joined = pd.read_csv('../data/joined.csv')

print 'splitting into train and test'

train = joined[joined['Hazard'] != -1]
test = joined[joined['Hazard'] == -1]

y = train['Hazard']
# X = train.drop(['Hazard', 'Id'], 1)
# X = train.drop(['Hazard', 'Id', 'T2_V10', 'T2_V7', 'T1_V13', 'T1_V10', 'tp_59', 'tp_84', 'global_mean', 'global_median', 'global_std'], 1)
# X_test = test.drop(['Hazard', 'Id', 'T2_V10', 'T2_V7', 'T1_V13', 'T1_V10', 'tp_59', 'tp_84', 'global_mean', 'global_median', 'global_std'], 1)
X = train.drop(['Hazard', 'Id'], 1)
X_test = test.drop(['Id', 'Hazard'], 1)

print "Let's apply log(1+x) transformation"
for column in X.columns:
  X[column] = X[column].apply(lambda x: math.log(1 + x), 1)
  X_test[column] = X_test[column].apply(lambda x: math.log(1 + x), 1)

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
test_size = 0.2

ind = 1
if ind == 1:
  n_iter = 5
  rs = ShuffleSplit(len(y), n_iter=n_iter, test_size=test_size, random_state=random_state)

  result = []
  for scale_pos_weight in [1]:
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
                params['scale_pos_weight'] = scale_pos_weight

                params_new = list(params.items())
                score = []
                # score_truncated_up = []
                # score_truncated_down = []
                score_truncated_both = []
                # score_truncated_both_round = []
                # score_truncated_both_int = []

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

                  preds = 0.5 * preds1 + 0.5 * preds2

                  tp = normalized_gini(y_test, preds)
                  score += [tp]
                  print tp

                sc = math.ceil(100000 * np.mean(score)) / 100000
                sc_std = math.ceil(100000 * np.std(score)) / 100000
                result += [(sc,
                            sc_std,
                            min_child_weight,
                            eta,
                            colsample_bytree,
                            max_depth,
                            subsample,
                            gamma,
                            n_iter,
                            params['objective'],
                            test_size,
                            scale_pos_weight)]

    result.sort()

    print
    print 'result'
    print result


elif ind == 2:
  xgtrain = xgb.DMatrix(X.values[offset:, :], label=y.values[offset:])
  xgval = xgb.DMatrix(X.values[:offset, :], label=y.values[:offset])
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
  model1 = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
  prediction_test_1 = model1.predict(xgtest, ntree_limit=model1.best_iteration)

  X_train = X.values[::-1, :]
  labels = y.values[::-1]

  xgtrain = xgb.DMatrix(X_train[offset:, :], label=labels[offset:])
  xgval = xgb.DMatrix(X_train[:offset, :], label=labels[:offset])

  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

  model2 = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)

  prediction_test_2 = model2.predict(xgtest, ntree_limit=model2.best_iteration)


  prediction_test = 0.5 * prediction_test_1 + 0.5 * prediction_test_2
  submission = pd.DataFrame()
  submission['Id'] = test['Id']
  submission['Hazard'] = prediction_test
  submission.to_csv("preds_on_test/xgbt.csv", index=False)

