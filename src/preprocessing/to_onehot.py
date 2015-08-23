from __future__ import division
__author__ = 'Vladimir Iglovikov'
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import OneHotEncoder

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

  features_num = ['T1_V1',
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
                  'T2_V15']

  train_new_temp = par[0]
  hold_new_temp = par[1]

  if len(par) == 3:
    test_new_temp = par[2]

  train_new = pd.DataFrame()
  test_new = pd.DataFrame()
  hold_new = pd.DataFrame()
  train_new['Id'] = train_new_temp['Id']
  train_new['Hazard'] = train_new_temp['Hazard']

  test_new['Id'] = test_new_temp['Id']

  test_new['Hazard'] = -1

  hold_new['Id'] = hold_new_temp['Id']
  hold_new['Hazard'] = hold_new_temp['Hazard']


  for feature in features_num:
    train_new[feature] = train_new_temp[feature].apply(lambda x: math.log(1 + x))
    test_new[feature] = test_new_temp[feature].apply(lambda x: math.log(1 + x))
    hold_new[feature] = hold_new_temp[feature].apply(lambda x: math.log(1 + x))


  train_new['type'] = 'train'
  test_new['type'] = 'test'
  hold_new['type'] = 'hold'

  joined = pd.concat([train_new, test_new, hold_new])
  joined = pd.concat([joined, pd.get_dummies(joined[features_cat])], 1)

  train_new = joined[joined['type'] == 'train']
  test_new = joined[joined['type'] == 'test']
  hold_new = joined[joined['type'] == 'hold']
  train_new.drop('type', 1, inplace=True)
  test_new.drop(['Hazard', 'type'], 1, inplace=True)

  hold_new.drop('type', 1, inplace=True)

  if len(par) == 2:
    return train_new, hold_new
  elif len(par) == 3:
    return train_new, hold_new, test_new