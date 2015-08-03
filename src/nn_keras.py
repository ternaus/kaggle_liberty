from __future__ import division

__author__ = 'Vladimir Iglovikov'

import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

joined = pd.read_csv('../data/joined.csv')

train = joined[joined['Hazard'] != -1]
test = joined[joined['Hazard'] == -1]

X = train.drop(['Id', 'Hazard'], 1)
X_test = test.drop(['Id', 'Hazard'], 1)

y = train['Hazard']

# Keras model
model = Sequential()
model.add(Dense(X.shape[1], 256))
model.add(Activation('sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(256, 256))
model.add(Activation('sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(256, 1))

model.compile(loss='mse', optimizer='rmsprop')

# train model, test on 15% hold out data
model.fit(X, y, batch_size=100,
        nb_epoch=100, verbose=2, validation_split=0.15)