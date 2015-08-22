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
preds = pd.read_csv('preds_on_hold/xgbt.csv')

def binar(x, a):
  if 1 < x < a:
    return 1
  elif a <= x < 2:
    return 2
  else:
    return x

x_list = np.arange(1, 2, 0.1)

y_list = []

for a in x_list:
  y_list += [normalized_gini(hold['Hazard'], map(lambda x: binar(x, a), preds['Hazard']))]

print x_list
print y_list
plot(x_list, y_list)
savefig('cuts.png')