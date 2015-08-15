from __future__ import division
__author__ = 'Vladimir Iglovikov'

'''
Here I will try to use xgb.
'''
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import ShuffleSplit
import numpy as np


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



xgtest = xgb.DMatrix(X_test)


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

rs = ShuffleSplit(len(y), n_iter=10, test_size=0.5, random_state=random_state)

result = []

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

            a_train = X.values[train_index]
            a_test = X.values[test_index]
            b_train = y.values[train_index]
            b_test = y.values[test_index]

            xgtrain = xgb.DMatrix(a_train[offset:, :], label=b_train[offset:])
            xgval = xgb.DMatrix(a_train[:offset, :], label=b_train[:offset])

            xtest = xgb.DMatrix(a_test, label=b_test)

            watchlist = [(xgtrain, 'train'), (xgval, 'val')]
            model = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
            preds = model.predict(xgtest, ntree_limit=model.best_iteration)


            score += [normalized_gini(b_test, preds)]

          result += [(np.mean(score), np.std(score), min_child_weight, eta, colsample_bytree, max_depth, subsample)]

result.sort()
print result

