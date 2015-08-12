from __future__ import division
__author__ = 'Vladimir Iglovikov'
from sklearn.linear_model import RidgeCV
from gini_normalized import normalized_gini
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
import numpy as np
from operator import itemgetter

'''
I will try to do prediction using linear regression, only using categorical variables.
'''

import pandas as pd

def report(grid_scores, n_top=10):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

train = pd.read_csv('../data/train.csv')
print train.info()
features = ['T1_V4',
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

gini = metrics.make_scorer(normalized_gini, greater_is_better=True)

X = pd.get_dummies(train[features])
y = train['Hazard']

params={'alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1, 10, 100, 1000]}
clf = Ridge(normalize=True)
ccv = GridSearchCV(clf, param_grid=params, scoring=gini, cv=10, verbose=10, n_jobs=-1)
ccv.fit(X, y)
print report(ccv.grid_scores_)

