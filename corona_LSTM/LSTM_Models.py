# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 22:18:31 2020

@author: ayanca
"""

# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.layers import Dense, LSTM, Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers import ConvLSTM2D
from keras.layers.convolutional import MaxPooling1D

def vanilla_LSTM(n_steps, n_features):
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse',  metrics=['acc'])    #model.compile(loss='mae', optimizer='adam')
    return model

def stacked_LSTM(n_steps, n_features):
    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['acc'])
    return model

def bidirectional_LSTM(n_steps, n_features):
    # define model
    model = Sequential()
    model.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['acc'])
    return model

def mul_layer_1(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(units = 92, return_sequences = True, input_shape = (n_steps, n_features)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 92, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 92, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 92, return_sequences = False))
    model.add(Dropout(0.2))
    
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam', loss = 'mse', metrics=['acc'])
    return model

def mul_layer_2(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(units = 100, return_sequences = True, input_shape = (n_steps, n_features)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 100, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 100, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 100, return_sequences = False))
    model.add(Dropout(0.2))
    
    model.add(Dense(units = 1))
    model.compile(optimizer = 'ADAM', loss = 'mse', metrics=['acc'])
    return model

def mul_layer_3(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (n_steps, n_features)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50, return_sequences = False))
    model.add(Dropout(0.2))
    
    model.add(Dense(units = 1))
    model.compile(optimizer = 'ADAM', loss = 'mse', metrics=['acc'])
    return model

'''
def cnn_LSTM(n_steps, n_features):
    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def conv_LSTM(n_steps, n_features):
    # define model
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(None, 1, n_steps, n_features)))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model
'''
