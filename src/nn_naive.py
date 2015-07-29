from __future__ import division

import pandas as pd
from pylab import *

from sklearn.cross_validation import train_test_split

def gini(solution, submission):
    df = zip(solution, submission, range(len(solution)))
    df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1, len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini


from sklearn import cross_validation

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import theano

from pylab import *
import seaborn as sns
import pandas as pd
import os
import xgboost as xgb
import numpy as np
import math


def float32(k):
    return np.cast['float32'](k)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()

class AdaptiveVariable(object):
    def __init__(self, name, start=0.03, stop=0.000001, inc=1.1, dec=0.5):
        self.name = name
        self.start, self.stop = start, stop
        self.inc, self.dec = inc, dec

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        if len(train_history) > 1:
            previous_valid = train_history[-2]['valid_loss']
        else:
            previous_valid = np.inf
        current_value = getattr(nn, self.name).get_value()
        if current_value < self.stop:
            raise StopIteration()
        if current_valid > previous_valid:
            getattr(nn, self.name).set_value(float32(current_value*self.dec))
        else:
            getattr(nn, self.name).set_value(float32(current_value*self.inc))

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


joined = pd.read_csv('../data/joined.csv')

train = joined[joined['Hazard'] != -1]

print train.shape
print list(train.columns)

features = [
  # 'Hazard',
  #           'Id',
            'T1_V1', 'T1_V10', 'T1_V13', 'T1_V14', 'T1_V17', 'T1_V2', 'T1_V3', 'T1_V6', 'T2_V1', 'T2_V10', 'T2_V11', 'T2_V12', 'T2_V14', 'T2_V15', 'T2_V2', 'T2_V3', 'T2_V4', 'T2_V6', 'T2_V7', 'T2_V8', 'T2_V9', 'tp_0', 'tp_1', 'tp_2', 'tp_3', 'tp_4', 'tp_5', 'tp_6', 'tp_7', 'tp_8', 'tp_9', 'tp_10', 'tp_11', 'tp_12', 'tp_13', 'tp_14', 'tp_15', 'tp_16', 'tp_17', 'tp_18', 'tp_19', 'tp_20', 'tp_21', 'tp_22', 'tp_23', 'tp_24', 'tp_25', 'tp_26', 'tp_27', 'tp_28', 'tp_29', 'tp_30', 'tp_31', 'tp_32', 'tp_33', 'tp_34', 'tp_35', 'tp_36', 'tp_37', 'tp_38', 'tp_39', 'tp_40', 'tp_41', 'tp_42', 'tp_43', 'tp_44', 'tp_45', 'tp_46', 'tp_47', 'tp_48', 'tp_49', 'tp_50', 'tp_51', 'tp_52', 'tp_53', 'tp_54', 'tp_55', 'tp_56', 'tp_57', 'tp_58', 'tp_59', 'tp_60', 'tp_61', 'tp_62', 'tp_63', 'tp_64', 'tp_65', 'tp_66', 'tp_67', 'tp_68', 'tp_69', 'tp_70', 'tp_71', 'tp_72', 'tp_73', 'tp_74', 'tp_75', 'tp_76', 'tp_77', 'tp_78', 'tp_79', 'tp_80', 'tp_81', 'tp_82', 'tp_83', 'tp_84']

random_state = 42

# train['Hazard'] = train['Hazard'].apply(lambda x: math.log(1 + x), 1)

X_train, X_test, y_train, y_test = train_test_split(train[features], train['Hazard'],
                                                    test_size=0.2,
                                                    random_state=random_state)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
y_mean = y_train.mean()
y_std = y_train.std()

X_test = scaler.transform(X_test)


from lasagne import layers
net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden0', layers.DenseLayer),
        ('dropout', DropoutLayer),
        ('hidden1', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, X_train.shape[1]),
    hidden0_num_units=200,  # number of units in hidden layer
    dropout_p=0.5,
    hidden1_num_units=200,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=1,  # 1 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.0001,
    update_momentum=0.9,
    eval_size=0.2,
    max_epochs=200,  # we want to train this many epochs
    verbose=1,
    regression=True,  # flag to indicate we're dealing with regression problem

    )

print X_train.shape
target = (y_train.values - y_mean) / y_std


net1.fit(X_train.astype(np.float32), target.astype(np.float32))


prediction = net1.predict(X_test)

# train = gl.SFrame('../data/train.csv')
#
#
#
# features = [
# #     'Id',
# #  'Hazard',
#  'T1_V1',
#  'T1_V2',
#  'T1_V3',
#  'T1_V4',
#  'T1_V5',
#  'T1_V6',
#  'T1_V7',
#  'T1_V8',
#  'T1_V9',
#  'T1_V10',
#  'T1_V11',
#  'T1_V12',
#  'T1_V13',
#  'T1_V14',
#  'T1_V15',
#  'T1_V16',
#  'T1_V17',
#  'T2_V1',
#  'T2_V2',
#  'T2_V3',
#  'T2_V4',
#  'T2_V5',
#  'T2_V6',
#  'T2_V7',
#  'T2_V8',
#  'T2_V9',
#  'T2_V10',
#  'T2_V11',
#  'T2_V12',
#  'T2_V13',
#  'T2_V14',
#  'T2_V15']
#
#
#
#
#
#
# ind = 1
# if ind == 1:
#   sf_train, sf_test = train.random_split(.8, seed=42)
#   model = gl.boosted_trees_regression.create(sf_train,
#                                            features=features,
#                                            target='Hazard',
#                                            validation_set=sf_test,
#                                            max_depth=4,
#                                            max_iterations=600,
#                                            step_size=0.01)
#
#   print normalized_gini(sf_test['Hazard'], model.predict(sf_test))
#
