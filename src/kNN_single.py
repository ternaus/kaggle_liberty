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
import time

joined = pd.read_csv('../data/joined.csv')
train = joined[joined['Hazard'] != -1]
test = joined[joined['Hazard'] == -1]

y = train['Hazard']
X = train.drop(['Hazard', 'Id'], 1)
X_test = test.drop(['Hazard', 'Id'], 1)

scaler = StandardScaler()

print 'scaling train'
X = scaler.fit_transform(X)

print 'scaling test'
X_test = scaler.transform(X_test)

clf = KNeighborsRegressor(n_neighbors=550)
print 'fitting'
clf.fit(X, y)
print 'predicting'
prediction = clf.predict(X_test)
submission = pd.DataFrame()
submission['Id'] = test['Id']
submission['Hazard'] = prediction
submission.to_csv('kNN/kNN_{timestamp}.csv'.format(timestamp=time.time()), index=False)

