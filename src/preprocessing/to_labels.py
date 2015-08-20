from __future__ import division
__author__ = 'Vladimir Iglovikov'
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

'''
This script will encode labels
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
    test_new = par[1]

  for feature in features_cat:
    le = LabelEncoder()

    le.fit(np.hstack([train_new[feature].values, hold_new[feature].values]))
    train_new[feature] = le.transform(train_new[feature])
    hold_new[feature] = le.transform(hold_new[feature])
    if len(par) == 3:
      test_new[feature] = le.transform(test_new[feature])

  if len(par) == 2:
    return train_new, hold_new
  elif len(par) == 3:
    return train_new, hold_new