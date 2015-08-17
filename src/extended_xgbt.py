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
# joined = pd.read_csv('../data/joined.csv')
#
# train = joined[joined['Hazard'] != -1]
# test = joined[joined['Hazard'] == -1]

train = pd.read_csv('../data/train.csv')

y = train['Hazard']
# X = train.drop(['Hazard', 'Id'], 1)
X = train.drop(['Id'], 1)
# X_test = test.drop(['Hazard', 'Id'], 1)


params = {
  'objective': 'reg:linear',
  # 'eta': 0.005,
  # 'min_child_weight': 6,
  # 'subsample': 0.7,
  # 'colsabsample_bytree': 0.7,
  'scal_pos_weight': 1,
  'silent': 1,
  # 'max_depth': 9
}

num_rounds = 10000
random_state = 42
offset = 5000

features = ['T1_V11',
            'T1_V12',
            'T1_V15',
            'T1_V16',
            'T1_V17',
            'T1_V4',
            'T1_V5',
            'T1_V6',
            'T1_V7',
            'T1_V8',
            'T1_V9',
            'T2_V11',
            'T2_V12',
            'T2_V13',
            'T2_V3',
            'T2_V5']

rs = ShuffleSplit(len(y), n_iter=10, test_size=0.5, random_state=random_state)

result = []

for train_index, test_index in rs:
  a_train = X.loc[train_index, :]
  a_test = X.loc[test_index, :]
  b_train = y.values[train_index]
  b_test = y.values[test_index]

  result = []
  for feature in features:
    grouped = a_train.groupby(feature)['Hazard'].agg([np.std, np.mean, np.median])
    grouped.reset_index(inplace=True)
    grouped.columns = [feature, feature + '_std', feature + '_mean', feature + '_median']
    result.append(grouped)

  for df in result:
    a_train = a_train.merge(df)
    a_test = a_test.merge(df)

  a_train.frop("Hazard", 1, inplace=True)
  a_test.frop("Hazard", 1, inplace=True)

  xgtrain = xgb.DMatrix(a_train.values[offset:, :], label=b_train[offset:])
  xgval = xgb.DMatrix(a_train.values[:offset, :], label=b_train[:offset])

  xgtest = xgb.DMatrix(a_test, label=b_test)
  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

  for min_child_weight in [9]:
    for eta in [0.005]:
      for colsample_bytree in [0.7]:
        for max_depth in [7]:
          for subsample in [0.7]:
            params['min_child_weight'] = min_child_weight
            params['eta'] = eta
            params['colsample_bytree'] = colsample_bytree
            params['max_depth'] = max_depth
            params['subsample'] = subsample
            score = []
            params_new = list(params.items())
            for train_index, test_index in rs:

              model = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
              preds = model.predict(xgtest, ntree_limit=model.best_iteration)


              score += [normalized_gini(b_test, preds)]

            result += [(np.mean(score), np.std(score), min_child_weight, eta, colsample_bytree, max_depth, subsample)]

result.sort()
print result
