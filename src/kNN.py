from __future__ import division
__author__ = 'Vladimir Iglovikov'

from operator import itemgetter
from sklearn import metrics
from gini_normalized import normalized_gini
import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
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

joined = pd.read_csv('../data/joined.csv')
train = joined[joined['Hazard'] != -1]

y = train['Hazard']
X = train.drop(['Hazard', 'Id'], 1)

clf = KNeighborsRegressor()

params = {'n_neighbors': [4, 5, 6, 7, 10, 15],
          'leaf_size': [10, 20, 30],
          }

ccv = GridSearchCV(clf, param_grid=params, scoring=gini, n_jobs=-1)
ccv.fit(X, y)

report(ccv.grid_scores_)