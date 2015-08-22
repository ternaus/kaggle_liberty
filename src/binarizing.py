from __future__ import division
__author__ = 'Vladimir Iglovikov'

'''
Initial Hazard values in train are integers [1: 69] with some missing values.
 For example 53-62 is not in the train => looks like an idea to put values in this region to
 53 or to 62, depending on the threashold
'''

import pandas as pd
from gini_normalized import normalized_gini
import numpy as np
from pylab import *

hold = pd.read_csv('../data/hold_new.csv')
preds = pd.read_csv('../preds_on_hold/xgbt.csv')

def binar(x, a):
  if 53 < x < a:
    return 53
  elif a <= x < 62:
    return 62
  else:
    return x

x_list = range(54, 62)

y_list = []

for a in x_list:
  y_list += [normalized_gini(hold['Hazard'], map(lambda x: binar(x, a)), preds['Hazard'])]

plot(x_list, y_list)
savefig('cuts.png')