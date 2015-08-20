from __future__ import division
__author__ = 'Vladimir Iglovikov'

'''
Cross validation for RF
'''
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import ShuffleSplit
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from preprocessing.to_labels import to_labels
from gini_normalized import normalized_gini

# joined = pd.read_csv('../data/joined.csv')
#
# train = joined[joined['Hazard'] != -1]
# test = joined[joined['Hazard'] == -1]

train = pd.read_csv('../data/train_new.csv')
hold = pd.read_csv('../data/hold_new.csv')

train, test = to_labels(train, hold)

y = train['Hazard']
X = train.drop(['Hazard', 'Id', 'T2_V10', 'T2_V7', 'T1_V13', 'T1_V10'], 1)
X_test = test.drop(['Hazard', 'Id', 'T2_V10', 'T2_V7', 'T1_V13', 'T1_V10'], 1)


random_state = 42

ind = 1

if ind == 1:
  rs = ShuffleSplit(len(y), n_iter=10, test_size=0.5, random_state=random_state)

  result = []

  for n_estimators in [10]:
    for min_samples_split in [2]:
      for max_features in [0.7]:
        for max_depth in [7]:
          for min_samples_leaf in [1]:
            score = []
            for train_index, test_index in rs:

              a_train = X.values[train_index]
              a_test = X.values[test_index]
              b_train = y.values[train_index]
              b_test = y.values[test_index]

              clf = RandomForestRegressor(n_estimators=n_estimators,
                                          min_samples_split=min_samples_split,
                                          max_features=max_features,
                                          max_depth=max_depth,
                                          min_samples_leaf=min_samples_leaf,
                                          n_jobs=-1,
                                          random_state=random_state)

              clf.fit(a_train, b_train)

              preds = clf.predict(a_test)

              score += [normalized_gini(b_test, preds)]

            result += [(np.mean(score), np.std(score), n_estimators, min_samples_split, min_samples_leaf, max_depth, max_features)]

  result.sort()
  print result

elif ind == 3:
  clf = RandomForestRegressor(n_estimators=100,
                              min_samples_split=2,
                              max_features=0.4,
                              max_depth=7,
                              min_samples_leaf=1,
                              n_jobs=-1,
                              random_state=random_state)
  clf.fit(X, y)
  prediction = clf.predict(X_test)
  print 'score on the hold = ', normalized_gini(test['Hazard'], prediction)

elif ind == 2:
  clf = RandomForestRegressor(n_estimators=100,
                              min_samples_split=2,
                              max_features=0.4,
                              max_depth=7,
                              min_samples_leaf=1,
                              n_jobs=-1,
                              random_state=random_state)
  clf.fit(X, y)
  prediction = clf.predict(X_test)
  submission = pd.DataFrame()
  submission['Id'] = test['Id']
  submission['Hazard'] = prediction
  submission.to_csv("preds_on_hold/RF.csv", index=False)