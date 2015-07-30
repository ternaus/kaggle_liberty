from __future__ import division
__author__ = 'Vladimir Iglovikov'

import math
import graphlab as gl

def XGB(X_train, y_train, X_test, y_test, uselog=False):
  '''

  :param X_train:
  :param y_train:
  :param X_test:
  :return:
  '''

  X = gl.SFrame(X_train)
  features = X.column_names()

  test = gl.SFrame(X_test)

  X['Hazard'] = y_train
  test['Hazard'] = y_test

  if uselog:
    X['Hazard'] = X['Hazard'].apply(lambda x: math.log(1 + x))
    test['Hazard'] = test['Hazard'].apply(lambda x: math.log(1 + x))


  model = gl.boosted_trees_regression.create(X,
                                             features=features,
                                             target='Hazard',
                                             validation_set=test,
                                             max_depth=7,
                                             max_iterations=800,
                                             step_size=0.01)
  return model.predict(test)