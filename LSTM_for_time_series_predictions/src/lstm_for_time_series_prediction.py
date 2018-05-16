#!/usr/bin/env python3
# coding: utf-8
# File: lstm_for_time_series_prediction.py
# Author: lxw
# Date: 5/15/18 3:21 PM
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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings


def plot_results(predicted_data, true_data):
    plt.plot(true_data, label="True Data")
    plt.plot(predicted_data, label="Prediction")
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    """
    :param predicted_data: list of list.
    :param true_data: shape: (413,)
    :param prediction_len: 
    :return: 
    """
    plt.plot(true_data, label="True Data")
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for _ in range(i * prediction_len)]
        plt.plot(padding + data, label="Prediction")
        plt.legend()
    plt.show()


def normalise_windows(window_data):
    # 经过该函数后，数据从str类型转为float类型
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data


def load_data(filename, window_size, normalise_window=True):
    # load the training data CSV into the appropriately shaped numpy array
    data = open(filename, "r").read()
    data_str_list = data.split("\n")  # len: 5001

    # print(data_str_list[-41:], len(data_str_list[-41:]))
    sequence_length = window_size + 1
    result = []
    for index in range(len(data_str_list) - sequence_length):  # range(4960)
        tmp_data_str_list = data_str_list[index: index + sequence_length]
        # draw_data_str_list(tmp_data_str_list)
        result.append(tmp_data_str_list)

    if normalise_window:
        # However running the adjusted returns of a stock index through a network would make the optimization
        # process shit itself and not converge to any sort of optimums for such large numbers.
        result = normalise_windows(result)
    else:
        result = [list(map(lambda x: float(x), data_str_list)) for data_str_list in result]


    # print(result[-1], len(result[-1]))
    result = np.array(result)  # result.shape: (4950, 51)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]  # train.shape: (4455, 51)
    np.random.shuffle(train)  # NOTE: inplace. `train` is shuffled.
    x_train = train[:, :-1]  # x_train.shape: (4455, 50)
    y_train = train[:, -1]  # y_train.shape: (4455,)    # (4455,)和(4455, 1)是不同的，前者是一维, 后者是二维
    x_test = result[int(row):, :-1]  # x_test.shape: (495, 50)
    y_test = result[int(row):, -1]  # y_test.shape: (495,)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # x_train.shape: (4455, 50, 1)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # x_test.shape: (495, 50, 1)
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

    model.add(LSTM(units=layers[1], return_sequences=False, name="lstm3"))  # TODO: return_sequences=False?
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
    predicted = model.predict(data)  # predicted.shape: (496, 1).
    predicted = np.reshape(predicted, (predicted.size,))  # predicted.shape: (496,)
    return predicted


def predict_sequence_full(model, data, window_size):
    """
    Shift the window by 1 new prediction each time, re-run predictions on new window
    
    :param data: data.shape: (496, 40, 1).
    """
    curr_frame = data[0]  # data.shape: (413, 40, 1). curr_frame.shape: (40, 1)
    predicted = []
    for i in range(len(data)):
        predict_result = model.predict(curr_frame[np.newaxis, :, :])  # curr_frame[np.newaxis, :, :].shape: (1, 40, 1)
        predicted.append(predict_result[0, 0])  # predict_result.shape: (1, 1)
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted


def predict_sequences_multiple(model, data, window_size, prediction_len):
    """
    Predict sequence of `prediction_len` steps before shifting prediction run forward by `prediction_len` steps
    
    :param data: data.shape: (413, 40, 1).
    """
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i * prediction_len]   # curr_frame.shape: (40, 1)
        predicted = []
        for _ in range(prediction_len):
            # curr_frame[np.newaxis, :, :].shape: (1, 40, 1)
            predict_result = model.predict(curr_frame[np.newaxis, :, :])
            predicted.append(predict_result[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def run():
    print("> Loading data...")
    X_train, y_train, X_test, y_test = load_data(filename="../data/sp500.csv", window_size=40, normalise_window=True)
    # X_train, y_train, X_test, y_test = load_data(filename="../data/sinwave.csv", window_size=40, normalise_window=False)
    # X_train.shape: ()

    print("> Data Loaded. Compiling...")
    layers = [51, 101, 1]
    model = build_model(layers=layers, input_shape=(X_train.shape[1], X_train.shape[2]))

    model.fit(X_train, y_train, batch_size=512, epochs=10, validation_split=0.05)
    print(model.summary())

    WINDOW_SIZE = 40
    PREDICTION_LEN = 20
    # predicted = predict_point_by_point(model, X_test)
    # predicted = predict_sequence_full(model=model, data=X_test, window_size=WINDOW_SIZE)  # NOTE: 这里的window_size要和load_data中的window_size一致
    predicted = predict_sequences_multiple(model=model, data=X_test, window_size=WINDOW_SIZE, prediction_len=PREDICTION_LEN)
    # print("len(predicted): {0}\npredicted:{1}\n".format(len(predicted), predicted))
    # print("len(true data): {0}\ntrue data:{1}".format(len(y_test), y_test))
    # print("predicted.shape:", predicted.shape)  # (496,)

    # plot_results(predicted, y_test)
    plot_results_multiple(predicted, y_test, prediction_len=PREDICTION_LEN)


if __name__ == "__main__":
    run()

