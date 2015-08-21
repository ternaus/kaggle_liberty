from __future__ import division
__author__ = 'Vladimir Iglovikov'
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import scipy

'''
This script will add statistical features computed on train to hold and test for categorical variables
and drop them after this
'''

def to_labels(par):
  features_cat = ["T1_V11",
                  "T1_V12",
                  "T1_V15",
                  "T1_V16",
                  "T1_V17",
                  "T1_V4",
                  "T1_V5",
                  "T1_V6",
                  "T1_V7",
                  "T1_V8",
                  "T1_V9",
                  "T2_V11",
                  "T2_V12",
                  "T2_V13",
                  "T2_V3",
                  "T2_V5"]
  train_new = par[0]
  hold_new = par[1]

  if len(par) == 3:
    test_new = par[2]


  #computing statistical features from averages from train for all categotical features
  result = []
  for feature in features_cat:
    grouped = train_new.groupby(feature)['Hazard'].agg([np.std, np.mean, np.median, scipy.stats.kurtosis, scipy.stats.skew])
    grouped.reset_index(inplace=True)
    grouped.columns = [feature,
                       feature + '_std',
                       feature + '_mean',
                       feature + '_median',
                       feature + '_kurtosis',
                       feature + '_skew']
    result.append(grouped)

  for df in result:
    train_new = train_new.merge(df)
    hold_new = hold_new.merge(df)
    if len(par) == 3:
      test_new = test_new.merge(df)


  #drop categorical variables
  train_new.drop(features_cat, inplace=True)
  test_new.drop(features_cat, inplace=True)
  hold_new.drop(features_cat, inplace=True)

  if len(par) == 2:
    return train_new, hold_new
  elif len(par) == 3:
    return train_new, hold_new, test_new