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

joined = pd.read_csv('../data/joined.csv')

train = joined[joined['Hazard'] != -1]
test = joined[joined['Hazard'] == -1]


y = train['Hazard']
X = train.drop(['Hazard', 'Id', 'T2_V10', 'T2_V7', 'T1_V13', 'T1_V10'], 1)
X_test = test.drop(['Hazard', 'Id', 'T2_V10', 'T2_V7', 'T1_V13', 'T1_V10'], 1)

random_state = 42

ind = 1

n_iter = 10
test_size = 0.2

if ind == 1:
  rs = ShuffleSplit(len(y), n_iter=10, test_size=0.2, random_state=random_state)

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

              clf1 = RandomForestRegressor(n_estimators=n_estimators,
                                          min_samples_split=min_samples_split,
                                          max_features=max_features,
                                          max_depth=max_depth,
                                          min_samples_leaf=min_samples_leaf,
                                          n_jobs=-1,
                                          random_state=random_state)

              clf1.fit(a_train, b_train)

              preds1 = clf1.predict(a_test)

              clf2 = RandomForestRegressor(n_estimators=n_estimators,
                                          min_samples_split=min_samples_split,
                                          max_features=max_features,
                                          max_depth=max_depth,
                                          min_samples_leaf=min_samples_leaf,
                                          n_jobs=-1,
                                          random_state=random_state)

              clf2.fit(a_train, np.log(b_train))

              preds2 = clf2.predict(a_test)

              preds = 0.5 * preds1 + 0.5 * np.exp(preds2)

              score += [normalized_gini(b_test, preds)]

            result += [(np.mean(score), np.std(score), n_estimators, min_samples_split, min_samples_leaf, max_depth, max_features, n_iter, test_size)]

  result.sort()
  print result


elif ind == 2:
  clf = RandomForestRegressor(n_estimators=100,
                              min_samples_split=2,
                              max_features=0.4,
                              max_depth=7,
                              min_samples_leaf=1,
                              n_jobs=-1,
                              random_state=random_state)
  clf.fit(X, y)

  prediction_test = clf.predict(X_test)
  submission = pd.DataFrame()
  submission['Id'] = test['Id']
  submission['Hazard'] = prediction_test
  submission.to_csv("preds_on_test/RF.csv", index=False)