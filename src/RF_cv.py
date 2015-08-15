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

joined = pd.read_csv('../data/joined.csv')

train = joined[joined['Hazard'] != -1]
test = joined[joined['Hazard'] == -1]

y = train['Hazard']
X = train.drop(['Hazard', 'Id'], 1)
X_test = test.drop(['Hazard', 'Id'], 1)

def gini(solution, submission):
    df = zip(solution, submission, range(len(solution)))
    df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1, len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini



random_state = 42

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
                                        n_jobs=-1)

            clf.fit(a_train, b_train)

            preds = clf.predict(a_test)


            score += [normalized_gini(b_test, preds)]

          result += [(np.mean(score), np.std(score), n_estimators, min_samples_split, min_samples_leaf, max_depth, max_features)]

result.sort()
print result

