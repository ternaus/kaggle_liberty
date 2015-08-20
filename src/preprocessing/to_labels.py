from __future__ import division
__author__ = 'Vladimir Iglovikov'
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

'''
This script will encode labels
'''

def to_labels(train, test):
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
  train_new = train
  test_new = test

  for feature in features_cat:
    le = LabelEncoder()
    le.fit(np.hstack([train_new[feature].values, test_new[feature].values]))
    train_new[feature] = le.fit_transform(train_new[feature])
    test_new[feature] = le.transform(test_new[feature])
  return train_new, test_new