from __future__ import division
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import theano
import numpy as np
from lasagne import layers
from lasagne.layers import DropoutLayer
from sklearn.preprocessing import StandardScaler
import math
from sklearn.neighbors import KNeighborsRegressor

__author__ = 'Vladimir Iglovikov'



def Ridge(X_train, y_train, X_test, y_test, uselog=False):
  '''

  :param X_train:
  :param y_train:
  :param X_test:
  :param y_test:
  :return:
  '''



  # clf = KNeighborsRegressor(n_neighbors=550)
  clf = Ridge(normalize=True, alpha=0.1)
  clf.fit(X_train, y_train)

  result = clf.predict(X_test)

  if uselog:
    result = map(lambda x: math.log(1 + x), result)

  return result

