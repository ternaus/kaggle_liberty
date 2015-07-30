from __future__ import division
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import theano
import numpy as np
from lasagne import layers
from lasagne.layers import DropoutLayer
from sklearn.preprocessing import StandardScaler
import math

__author__ = 'Vladimir Iglovikov'

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


def NN(X_train, y_train, X_test, y_test, uselog=False):
  '''

  :param X_train:
  :param y_train:
  :param X_test:
  :param y_test:
  :return:
  '''

  net1 = NeuralNet(
      layers=[  # three layers: one hidden layer
          ('input', layers.InputLayer),
          ('dropout1', DropoutLayer),
          ('hidden0', layers.DenseLayer),
          ('dropout2', DropoutLayer),
          ('hidden1', layers.DenseLayer),
          ('output', layers.DenseLayer),
          ],
      # layer parameters:
      input_shape=(None, X_train.shape[1]),
      dropout1_p=0.1,
      hidden0_num_units=200,  # number of units in hidden layer
      dropout2_p=0.3,
      hidden1_num_units=200,  # number of units in hidden layer
      output_nonlinearity=None,  # output layer uses identity function
      output_num_units=1,  # 1 target values

      # optimization method:
      update=nesterov_momentum,
      # update_learning_rate=0.001,
      # update_momentum=0.9,
      update_momentum=theano.shared(float32(0.9)),
      eval_size=0.2,
      max_epochs=1000,  # we want to train this many epochs
      update_learning_rate=theano.shared(float32(0.03)),
      verbose=1,
      regression=True,  # flag to indicate we're dealing with regression problem
      on_epoch_finished=[
                      AdaptiveVariable('update_learning_rate', start=0.001, stop=0.00001),
                      AdjustVariable('update_momentum', start=0.9, stop=0.999),
                      EarlyStopping(),
                  ]
      )
  scaler = StandardScaler()
  print X_train.shape
  print X_test.shape

  X = scaler.fit_transform(X_train).astype(np.float32)
  test = scaler.transform(X_test).astype(np.float32)

  y = y_train.values
  y.shape = (y.shape[0], 1)

  y_mean = y.mean()
  y_std = y.std()
  target = (y - y_mean) / y_std

  net1.fit(X, target.astype(np.float32))

  def helper(x):
    return (x * y_std) + y_mean

  result = net1.predict(test)
  result = np.reshape(result, len(y_test))

  result = map(helper, result)

  if uselog:
    result = map(lambda x: math.log(1 + x), result)

  return result

