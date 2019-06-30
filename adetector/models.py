"""This module defines all the model functions"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.callbacks import ModelCheckpoint

from .config import N_MFCC, N_TIMEBINS, N_FEATURES


def create_CNN_model(quiet=False):
    '''Creates a model obejct with a 2D input
       of shape (n_mfcc, n_timebins,1)'''
    model = Sequential()  # create a model instance

    # add model layers
    model.add(Conv2D(16, (3, 3), strides=(1, 1), activation='relu',
                     padding='same', input_shape=(N_MFCC, N_TIMEBINS, 1)))
    model.add(Conv2D(16, (3, 3), strides=(1, 1), activation='relu',
                     padding='same'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    if not quiet:
        model.summary()

    return model


def create_NN_model(quiet=False):
    '''Create a model obejct with an input of length n_features'''
    model = Sequential()  # create a model instance

    # add model layers
    model.add(Dense(256, activation='relu', input_shape=(N_FEATURES,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    if not quiet:
        model.summary()

    return model
