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
  # 'scal_pos_weight': 1,
  'silent': 0,
  # 'max_depth': 9
}

num_rounds = 10000
random_state = 42
params = list(params.items())


rs = ShuffleSplit(len(y), n_iter=5, test_size=0.2, random_state=random_state)

score = []
for train_index, test_index in rs:
    a_train = X.values[train_index]
    a_test = X.values[test_index]
    b_train = y.values[train_index]
    b_test = y.values[test_index]

    xgtrain = xgb.DMatrix(a_train, label=b_train)
    xgval = xgb.DMatrix(a_test, label=b_test)


    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
    preds = model.predict(xgval, ntree_limit=model.best_iteration)


    score += [normalized_gini(b_test, preds)]

print 'score'
print np.mean(score), np.std(score)
