from __future__ import division

__author__ = 'Vladimir Iglovikov'

'''
XGB works really well so far => I will try to use it as a metaestimator

[1] train best XGB model on train_new
[2] train best RF model on train_new
[3] Add predictions to hold_new
[4] Train xgb on the hold and predict on test
'''