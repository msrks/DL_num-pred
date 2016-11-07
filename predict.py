# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import numpy as np

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

OUTNAME = "conv32conv64fc1024fc256fc10.mdl"

### Build CNN
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 1024, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')
model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir="logs")

### Load Model
model.load("models/"+OUTNAME)
X = np.zeros((1,784))
X = X.reshape([-1, 28, 28, 1])
pred = model.predict(X)[0]
print(pred)
print(np.argmax(pred))
