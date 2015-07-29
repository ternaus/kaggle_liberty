__author__ = 'Vladimir Iglovikov'

import graphlab as gl

joined = gl.SFrame('../data/joined.csv')

train = joined[joined['Hazard'] != -1]

sf_train, sf_test = train.random_split(0.8, seed=42)

sf_train.save('../data/train1.csv')
sf_test.save('../data/hold.csv')