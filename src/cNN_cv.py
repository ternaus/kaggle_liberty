
from __future__ import division
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
# from nolearn.lasagne.base import TrainSplit
from sklearn.cross_validation import ShuffleSplit
import theano
import numpy as np
from lasagne import layers
from lasagne.layers import DropoutLayer
from sklearn.preprocessing import StandardScaler
import math
import pandas as pd
__author__ = 'Vladimir Iglovikov'
from gini_normalized import normalized_gini

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
test = joined[joined['Hazard'] == -1]

y_train = train['Hazard']
X_train = train.drop(['Hazard', 'Id', 'T2_V10', 'T2_V7', 'T1_V13', 'T1_V10', 'tp_84', 'tp_59'], 1)
X_test = test.drop(['Hazard', 'Id'], 1)


net1 = NeuralNet(
      input_shape=(None, 1, 10, 10),
  layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
      conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
      conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
      conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
      hidden4_num_units=500,
      hidden5_num_units=500,

      output_nonlinearity=None,  # output layer uses identity function
      output_num_units=1,  # 1 target values

      # optimization method:
      update=nesterov_momentum,
      # update_learning_rate=0.001,
      # update_momentum=0.9,
      update_momentum=theano.shared(float32(0.9)),
      # train_split=TrainSplit(eval_size=0.2),

      eval_size=0.2,
      max_epochs=100,  # we want to train this many epochs
      update_learning_rate=theano.shared(float32(0.03)),
      verbose=1,
      regression=True,  # flag to indicate we're dealing with regression problem
      on_epoch_finished=[
                      AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
                      AdjustVariable('update_momentum', start=0.9, stop=0.999),
                      # AdaptiveVariable('update_learning_rate', start=0.03, stop=0.0001),
                      # AdjustVariable('update_momentum', start=0.9, stop=0.999),
                      EarlyStopping(),
                  ]
      )


scaler = StandardScaler()
print X_train.shape
# print X_test.shape
random_state = 42

rs = ShuffleSplit(len(y_train), n_iter=1, test_size=0.5, random_state=random_state)

score = []

for train_index, test_index in rs:
  a_train = X_train.values[train_index]
  a_test = X_train.values[test_index]
  b_train = y_train.values[train_index]
  b_test = y_train.values[test_index]

  X = scaler.fit_transform(a_train).astype(np.float32)
  X_reshaped = X.reshape(-1, 1, 10, 10)
  test_reshaped = test.reshape(-1, 1, 10, 10)
  # test = scaler.transform(a_test).astype(np.float32)

  y = b_train[:]
  y.shape = (y.shape[0], 1)

  y_mean = y.mean()
  y_std = y.std()
  target = (y - y_mean) / y_std

  net1.fit(X_reshaped, target.astype(np.float32))

  def helper(x):
    return (x * y_std) + y_mean

  result = net1.predict(test_reshaped)

  result = np.reshape(result, len(b_test))
  result = map(helper, result)
  score += [normalized_gini(b_test, result)]


print np.mean(score), np.std(score)