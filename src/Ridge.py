from __future__ import division
__author__ = 'Vladimir Iglovikov'

from operator import itemgetter
from sklearn import metrics
from gini_normalized import normalized_gini
import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import time

def report(grid_scores, n_top=10):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

gini = metrics.make_scorer(normalized_gini, greater_is_better=True)


random_state = 42

joined = pd.read_csv('../data/joined_extended.csv')
train = joined[joined['Hazard'] != -1]
test = joined[joined['Hazard'] == -1]

y = train['Hazard']
X = train.drop(['Hazard', 'Id'], 1)
X_test = test.drop(['Hazard', 'Id'], 1)

ind = 1
if ind == 1:
  params={'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1, 10, 100, 1000]}

  clf = Ridge(normalize=True)
  ccv = GridSearchCV(clf, param_grid=params, scoring=gini, cv=10, verbose=10, n_jobs=-1)

  ccv.fit(X, y)

  report(ccv.grid_scores_)
elif ind == 2:
  clf = Ridge(normalize=True, alpha=0.1)
  clf.fit(X, y)
  prediction = clf.predict(X_test)
  submission = pd.DataFrame()
  submission['Id'] = test['Id']
  submission['Hazard'] = prediction
  submission.to_csv('predictions/Ridge_{timestamp}.csv'.format(timestamp=time.time()), index=False)