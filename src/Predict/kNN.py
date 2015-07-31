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



def kNN(X_train, y_train, X_test, y_test, uselog=False):
  '''

  :param X_train:
  :param y_train:
  :param X_test:
  :param y_test:
  :return:
  '''

  scaler = StandardScaler()
  print X_train.shape
  print X_test.shape

  X = scaler.fit_transform(X_train)
  test = scaler.transform(X_test)

  clf = KNeighborsRegressor()

  clf.fit(X, y_train)

  result = clf.predict(test)

  if uselog:
    result = map(lambda x: math.log(1 + x), result)

  return result

