__author__ = 'Vladimir Iglovikov'

'''
I would like to check, how will number of PCA components will affect score for RandomForest
'''

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
import numpy as np
from gini_normalized import normalized_gini
from sklearn import metrics

train = pd.read_csv('../data/train.csv')

cat_features = ['T1_V4',
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

num_features = [
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

y = train['Hazard']

scaler = StandardScaler(with_std=False)

X_cat_original = pd.get_dummies(train[cat_features])

X_cat_original = scaler.fit_transform(X_cat_original)

X_num = train[num_features]

scores_mean = []
score_err = []
gini = metrics.make_scorer(normalized_gini, greater_is_better=True)

for n in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
  pca = PCA(n_components=n)

  X_cat = pca.fit_transform(X_cat_original)

  X = pd.concat([X_cat, X_num], 1)
  print X.shape
  clf = RandomForestRegressor(n_estimators=1000, max_features='sqrt', n_jobs=-1)
  scores = cross_validation.cross_val_score(clf, X, y, cv=10, scoring=gini)
  scores_mean += [np.mean(scores)]
  score_err += [2 * np.std(scores)]

print 'mean = ', scores_mean
print 'err = ', score_err