# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import flask
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

### Flask App
app = flask.Flask(__name__)

@app.route("/")
def main():
    return flask.render_template("test.html")

@app.route("/score", methods=["POST"])
def score():
    data = flask.request.json

    current_image = np.zeros((1,784))
    for i in data["example"]:
        if i >= 0 and i < 784:
            current_image[0][i] = 1
    X = current_image.reshape([-1, 28, 28, 1])
    pred = model.predict(X)[0]
    #pred = np.random.random(10)/5
    #pred = pred.tolist()
    #pred = [0.001, 0.002, 0.005, 0.4, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2]
    results = {"pred": pred}
    return flask.jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
