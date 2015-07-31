from __future__ import division
__author__ = 'Vladimir Iglovikov'

from operator import itemgetter
from sklearn import metrics
from gini_normalized import normalized_gini
import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

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

joined = pd.read_csv('../data/joined.csv')
train = joined[joined['Hazard'] != -1]

y = train['Hazard']
X = train.drop(['Hazard', 'Id'], 1)

clf = RandomForestRegressor(n_jobs=-1, random_state=random_state)

params = {'n_estimators' : [100, 200, 500, 1000],
          'max_depth': range(4, 12),
          'min_samples_split': range(2, 5),
          'min_samples_leaf': range(1, 4),
          'bootstrap': [True, False],

          }

ccv = RandomizedSearchCV(clf, param_distributions=params, scoring=gini, cv=5, verbose=10)
ccv.fit(X, y)

report(ccv.grid_scores_)