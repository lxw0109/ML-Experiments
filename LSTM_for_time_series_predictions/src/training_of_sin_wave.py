#!/usr/bin/env python3
# coding: utf-8
# File: training_of_sin_wave.py
# Author: lxw
# Date: 5/15/18 3:33 PM
"""
References:
1. [LSTM Neural Network for Time Series Prediction](http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction)
"""


import matplotlib.pyplot as plt
import numpy as np
import os
import time
import warnings

from keras import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")    # Hide messy Numpy warnings


def plot_results(predicted_data, true_data):
    true_data = list(map(lambda x: float(x), true_data))
    plt.plot(true_data, label="True Data")
    predicted_data = list(map(lambda x: float(x), predicted_data))
    plt.plot(predicted_data, label="Prediction")
    plt.legend()
    plt.show()


def load_data(filename, window_size):
    # load the training data CSV into the appropriately shaped numpy array
    data = open(filename, "r").read()
    data_str_list = data.split("\n")    # len: 5001

    # print(data_str_list[-41:], len(data_str_list[-41:]))
    sequence_length = window_size + 1
    result = []
    for index in range(len(data_str_list) - sequence_length):     # range(4960)
        tmp_data_str_list = data_str_list[index: index + sequence_length]
        # draw_data_str_list(tmp_data_str_list)
        result.append(tmp_data_str_list)

    # print(result[-1], len(result[-1]))
    result = np.array(result)    # result.shape: (4950, 51)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]    # train.shape: (4455, 51)
    np.random.shuffle(train)    # NOTE: inplace. `train` is shuffled.
    x_train = train[:, :-1]     # x_train.shape: (4455, 50)
    y_train = train[:, -1]    # y_train.shape: (4455,)    # (4455,)和(4455, 1)是不同的，前者是一维, 后者是二维
    x_test = result[int(row):, :-1]    # x_test.shape: (495, 50)
    y_test = result[int(row):, -1]    # y_test.shape: (495,)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))    # x_train.shape: (4455, 50, 1)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))      # x_test.shape: (495, 50, 1)
    # x_train & x_test: ndarray of ndarray of ndarray of str.

    return x_train, y_train, x_test, y_test


def build_model(layers, input_shape):
    """
    :param layers: [# of neurons in input layer, # of neurons in hidden layer, # of neurons in output layer]
    :param input_shape: 
    :return: 
    """
    model = Sequential()

    # The way Keras LSTM layers work is by taking in a numpy array of 3 dimensions (N, W, F) where N is the
    # number of training sequences, W is the sequence length and F is the number of features of each sequence.
    # model.add(LSTM(input_dim=layers[0], output_dim=layers[1], return_sequences=True))    # Deprecated
    # model.add(LSTM(input_shape=(None, layers[0]), units=layers[1], return_sequences=True))
    model.add(LSTM(input_shape=input_shape, units=layers[0], return_sequences=True, name="lstm1"))
    model.add(Dropout(0.2, name="dropout2"))

    model.add(LSTM(units=layers[1], return_sequences=False, name="lstm3"))    # TODO: return_sequences=False?
    model.add(Dropout(0.2, name="dropout4"))

    # model.add(Dense(output_dim=layers[3]))    # Deprecated
    model.add(Dense(units=layers[2], name="dense5"))
    model.add(Activation("linear", name="activation6"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time: ", time.time() - start)

    return model


def predict_point_by_point(model, data):
    """
    Predict each timestep given the last sequence of **true data**, in effect **only predicting 1 step ahead** each time
    
    :param model: 
    :param data: data.shape: (496, 40, 1).
    :return: 
    """
    predicted = model.predict(data)    # predicted.shape: (496, 1).
    predicted = np.reshape(predicted, (predicted.size,))    # predicted.shape: (496,)
    return predicted


def run():
    print("> Loading data...")
    # def load_data(filename, window_size, normalise_window=False):
    X_train, y_train, X_test, y_test = load_data("../data/sinwave.csv", 40)
    # X_train.shape: (4464, 40, 1)

    print("> Data Loaded. Compiling...")
    layers = [51, 101, 1]
    model = build_model(layers=layers, input_shape=(X_train.shape[1], X_train.shape[2]))

    model.fit(X_train, y_train, batch_size=512, epochs=1, validation_split=0.05)
    print(model.summary())

    predicted = predict_point_by_point(model, X_test)
    print("predicted:", predicted)
    print("predicted.shape:", predicted.shape)    # (496,)

    plot_results(predicted, y_test)


if __name__ == "__main__":
    run()
