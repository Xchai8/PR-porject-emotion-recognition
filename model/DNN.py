import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense, LSTM

def ann_estimator(data, label, lr, batch_size, epochs):
    
    scaler = StandardScaler()
    data_scl = scaler.fit_transform(data)
    data_ts = tf.constant(data_scl, dtype=tf.float16)

    model = keras.models.Sequential([keras.layers.Input((310)),
                                      keras.layers.Dense(128, kernel_initializer='he_normal'),
                                      keras.layers.ReLU(),
                                      keras.layers.Dense(64, kernel_initializer='he_normal'),
                                      keras.layers.ReLU(),
                                      keras.layers.Dense(32, kernel_initializer='he_normal'),
                                      keras.layers.ReLU(),
                                      keras.layers.Dense(3, activation='softmax')])

    optimizer = keras.optimizers.Nadam(learning_rate=lr)
    loss = 'sparse_categorical_crossentropy'
    earlystop = keras.callbacks.EarlyStopping(monitor='loss', 
                                              patience=5)

    model.compile(loss=loss, 
                  optimizer=optimizer,
                  metrics=['accuracy'])

    history = model.fit(data_ts, label, epochs=epochs, batch_size=batch_size, callbacks=[earlystop])

    return model, history


def cnn_estimator(data, label, lr, batch_size, epochs):
    
    scaler = StandardScaler()
    data_scl = scaler.fit_transform(data)
    data_ts = tf.constant(data_scl, dtype=tf.float16)

    model = keras.models.Sequential()
    # Convolutional layers
    model.add(Conv1D(32, 3, activation='relu', input_shape=(310,1)))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.25))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.25))
    # Global average pooling layer
    model.add(GlobalAveragePooling1D())
    # Fully connected layers
    model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    optimizer = keras.optimizers.Nadam(learning_rate=lr)
    loss = 'sparse_categorical_crossentropy'
    earlystop = keras.callbacks.EarlyStopping(monitor='loss', 
                                              patience=5)

    model.compile(loss=loss, 
                  optimizer=optimizer,
                  metrics=['accuracy'])

    history = model.fit(data_ts, label, epochs=epochs, batch_size=batch_size, callbacks=[earlystop])

    return model, history


def lstm_estimator(data, label, lr, batch_size, epochs):

    scaler = StandardScaler()
    data_scl = scaler.fit_transform(data)
    data_ts = tf.constant(data_scl, dtype=tf.float16)

    model = keras.models.Sequential()
    # LSTM layers
    model.add(LSTM(64, input_shape=(310, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    # Fully connected layers
    model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax')) 

    optimizer = keras.optimizers.Nadam(learning_rate=lr)
    loss = 'sparse_categorical_crossentropy'
    earlystop = keras.callbacks.EarlyStopping(monitor='loss', 
                                              patience=5)

    model.compile(loss=loss, 
                  optimizer=optimizer,
                  metrics=['accuracy'])

    history = model.fit(data_ts, label, epochs=epochs, batch_size=batch_size, callbacks=[earlystop])

    return model, history