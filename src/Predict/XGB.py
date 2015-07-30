from __future__ import division
__author__ = 'Vladimir Iglovikov'

import graphlab as gl

def XGB(X_train, y_train, X_test, y_test):
  '''

  :param X_train:
  :param y_train:
  :param X_test:
  :return:
  '''
  features = X_train.columns
  X = X_train
  X.loc[:, 'Hazard'] = y_train
  test = X_test

  test = gl.SFrame(test)
  test['Hazard'] = y_test

  model = gl.boosted_trees_regression.create(gl.SFrame(X_train),
                                             features=features,
                                             target='Hazard',
                                             validation_set=test,
                                             max_depth=7,
                                             max_iterations=800,
                                             step_size=0.01)
  return model.predict(test)