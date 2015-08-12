from __future__ import division
__author__ = 'Vladimir Iglovikov'


'''
Let's try this trick.
I will use Ridge on categorical variables, add their prediction to numerical and use XGB on this.
'''
from sklearn import cross_validation
import pandas as pd
from sklearn.linear_model import Ridge
import xgboost as xgb
from gini_normalized import normalized_gini
import numpy as np

n_iter = 2
random_state = 42

train = pd.read_csv('../data/train.csv')

rs = cross_validation.StratifiedKFold(y, n_folds=n_iter, shuffle=True, random_state=random_state)

features_cat = ['T1_V4',
            'T1_V5',
            'T1_V6',
            'T1_V7',
            'T1_V8',
            'T1_V9',
            'T1_V11',
            'T1_V12',
            'T1_V15',
            'T1_V16',
            'T1_V17',
            'T2_V3',
            'T2_V5',
            'T2_V11',
            'T2_V12',
            'T2_V13']

features_num = [
  'T1_V1',
  'T1_V2',
  'T1_V3',
  'T1_V10',
  'T1_V13',
  'T1_V14',
  'T2_V1',
  'T2_V2',
  'T2_V4',
  'T2_V6',
  'T2_V7',
  'T2_V8',
  'T2_V9',
  'T2_V10',
  'T2_V14',
  'T2_V15'
]

X_cat = pd.get_dummies(train[features_cat])
X_num = train[features_num]

y = train['Hazard']

num_rounds = 10000
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

score = []
for train_index, test_index in rs:
  Xc_train = X_cat.values[train_index]
  Xc_test = X_cat.values[test_index]
  y_train = y.values[train_index]
  y_test = y.values[test_index]
  clf_cat = Ridge(normalize=True, alpha=0.1)
  clf_cat.fit(Xc_train, y_train)
  prediction_cat_test = clf_cat.predict(Xc_test)
  prediction_cat_train = clf_cat.predict(Xc_train)
  Xn_train = X_num.values[train_index]
  Xn_test = X_num.values[test_index]
  Xn_train = pd.DataFrame(Xn_train)
  Xn_test = pd.DataFrame(Xn_test)
  Xn_train['cat'] = prediction_cat_train
  Xn_test['cat'] = prediction_cat_test
  xgtrain = xgb.DMatrix(Xn_train, label=y_train)
  xgval = xgb.DMatrix(Xn_test, label=y_test)
  watchlist = [(xgtrain, 'train'), (xgval, 'val')]
  model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=200)
  preds = model.predict(xgval, ntree_limit=model.best_iteration)
  score += [normalized_gini(y_test, preds)]

print np.mean(score), np.std(score)