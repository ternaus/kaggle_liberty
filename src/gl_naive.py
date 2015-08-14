from __future__ import division

import pandas as pd
from pylab import *
import graphlab as gl

def gini(solution, submission):
    df = zip(solution, submission, range(len(solution)))
    df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini

train = gl.SFrame('../data/train.csv')



features = [
#     'Id',
#  'Hazard',
 'T1_V1',
 'T1_V2',
 'T1_V3',
 'T1_V4',
 'T1_V5',
 'T1_V6',
 'T1_V7',
 'T1_V8',
 'T1_V9',
 'T1_V10',
 'T1_V11',
 'T1_V12',
 'T1_V13',
 'T1_V14',
 'T1_V15',
 'T1_V16',
 'T1_V17',
 'T2_V1',
 'T2_V2',
 'T2_V3',
 'T2_V4',
 'T2_V5',
 'T2_V6',
 'T2_V7',
 'T2_V8',
 'T2_V9',
 'T2_V10',
 'T2_V11',
 'T2_V12',
 'T2_V13',
 'T2_V14',
 'T2_V15']






ind = 1
if ind == 1:
  sf_train, sf_test = train.random_split(0.8, seed=42)
  
  model = gl.boosted_trees_regression.create(sf_train, 
                                           features=features, 
                                           target='Hazard', 
                                           validation_set=sf_test,
                                           max_depth=4,
                                           max_iterations=600,
                                           step_size=0.01)

  print normalized_gini(sf_test['Hazard'], model.predict(sf_test))

