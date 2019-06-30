"""This module defines all the functions for training a
   a model and assesing performance """
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import keras
from keras.callbacks import ModelCheckpoint

from .models import create_CNN_model
from .utils import plot_confusion_matrix
from .DataGenerator import DataGenerator_Sup
from . import config


def list_data():
    '''
    lists all data present in the directories defined in config.py
    returns a list of paths to positive files, music files and podcast files
    '''
    pos_files = []  # a list of paths to positive examples
    for r, d, f in os.walk(config.AD_FOLDER):
        for filename in f:
            if '.mp3' in filename:
                pos_files.append(os.path.join(r, filename))

    music_files = []  # a list of paths to negative music examples
    for r, d, f in os.walk(config.MUSIC_FOLDER):
        for filename in f:
            if '.mp3' or '.au' in filename:
                music_files.append(os.path.join(r, filename))

    podcast_files = []  # a list of paths to negative podcast examples
    for r, d, f in os.walk(config.PODCAST_FOLDER):
        for filename in f:
            if '.wav' in filename:
                podcast_files.append(os.path.join(r, filename))

    neg_files = music_files + podcast_files  # a list of all negative paths
    n_pos_files = len(pos_files)  # number of positive files
    n_neg_files = len(neg_files)  # number of negative files

    print('There are ' + str(n_pos_files) + ' positive examples')
    print('There are ' + str(len(music_files)) + ' music examples')
    print('There are ' + str(len(podcast_files)) + ' podcast examples')

    # compute the amount of data in minutes
    pos_minutes = round(config.AD_FILE_DURATION*n_pos_files, 2)
    neg_minutes = round(config.MUSIC_FILE_DURATION*len(music_files) +
                        config.PODCAST_FILE_DURATION*len(podcast_files), 2)
    print('--------------------------------')
    print('In total, ' + str(pos_minutes) + ' minutes of positive and ' +
          str(neg_minutes) + ' minutes of negative')

    return pos_files, music_files, podcast_files


def create_data_generators(pos_files, neg_files, data_minutes=1000,
                           train_fraction=0.9, neg_type=False):
    '''
    creates a balanced data generators for training and testing models
    inputs:
    ------
    pos_files - a list of paths to all the positive files
    neg_files - a list of paths to all the negative files
    data_minutes - the total amount of data to be used in minutes
    train_fraction - the fraction of the data that would be used for training
    neg_type - defines which negatives are used. 0 = music, 1 = podcasts
    outputs:
    -------
    train_generator - a keras class for generating training data
    test_generator -  a keras class for generating testing data
    '''
    pos_minutes2use = int(data_minutes/2.0)
    neg_minutes2use = int(data_minutes/2.0)

    # calculate train and test minutes splitting
    # train
    # number of pos audio training minutes
    pos_train_minutes = round(train_fraction*pos_minutes2use, 2)
    # number of minutes for music training
    neg_train_minutes = round(train_fraction*neg_minutes2use, 2)
    # test
    pos_test_minutes = pos_minutes2use - pos_train_minutes
    neg_test_minutes = neg_minutes2use - neg_train_minutes

    print('Using ' + str(pos_train_minutes) +
          ' minutes of positives for training and ' +
          str(pos_test_minutes) + ' for testing')
    print('Using ' + str(neg_train_minutes) +
          ' minutes of negatives for training and ' +
          str(neg_test_minutes) + ' for testing')

    # calculate number of files
    # train
    n_pos_train_files = int(pos_train_minutes/config.AD_FILE_DURATION)
    n_pos_test_files = int(pos_test_minutes/config.AD_FILE_DURATION)
    if neg_type:
        n_neg_train_files = int(neg_train_minutes/config.PODCAST_FILE_DURATION)
        n_neg_test_files = int(neg_test_minutes/config.PODCAST_FILE_DURATION)
    else:
        n_neg_train_files = int(neg_train_minutes/config.MUSIC_FILE_DURATION)
        n_neg_test_files = int(neg_test_minutes/config.MUSIC_FILE_DURATION)
    
    print('---------------------------------------------------------------')
    print('This translates into:')
    print(str(n_pos_train_files) + ' poitive files for training ' + 'and ' +
          str(n_pos_test_files) + ' for testing')
    if neg_type:
        print(str(n_neg_train_files) + ' podcasts files for training ' +
              'and ' + str(n_neg_test_files) + ' for testing')
    else:
        print(str(n_neg_train_files) + ' music files for training ' +
              'and ' + str(n_neg_test_files) + ' for testing')

    assert len(neg_files) >= n_neg_train_files + \
        n_neg_test_files, 'There are not enough negative files for that!'

    '''Shuffling data and creating train and test list and'''
    train_files = []  # a list of training files
    test_files = []  # a list of test files

    # shuffle files
    np.random.seed(1)
    np.random.shuffle(pos_files)
    np.random.seed(2)
    np.random.shuffle(neg_files)

    '''Collect a balanced list of files + add labels'''
    # Training list
    for f in pos_files[:n_pos_train_files]:
        train_files.append([f, 1])
    for f in neg_files[:n_neg_train_files]:
        train_files.append([f, 0])

    # Test list
    for f in pos_files[n_pos_train_files:n_pos_train_files + n_pos_test_files]:
        test_files.append([f, 1])
    for f in neg_files[n_neg_train_files:n_neg_train_files + n_neg_test_files]:
        test_files.append([f, 0])

    train_generator = DataGenerator_Sup(train_files, dataset='train', CNN=True)
    test_generator = DataGenerator_Sup(test_files, dataset='test', CNN=True)

    return train_generator, test_generator


def train_CNN_model(train_generator, epochs=10,
                    path_to_ckpt_file='model1.hdf5'):
    '''
    trains a CNN model, saves a chekpoint of the weights after each epoch and
    returns a history dictionary with values of loss and accuracy per epoch
    inputs:
    ------
    train_generator - a keras data generator object, an output of
                      the create_data_generators function
    epoch - number of epochs to train on
    path_to_ckpt_file - a path to a file which would hold the trained weights

    outputs:
    -------
    history - a history dictionary, history['loss'] and history['acc'] holds
              the loss and accuracy at the end of each epoch
    '''
    model = create_CNN_model()
    checkpoint = ModelCheckpoint(path_to_ckpt_file)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    H = model.fit_generator(generator=train_generator,
                            epochs=epochs, callbacks=[checkpoint])
    return H.history


def evaluate_model(model_weights_path, test_generator, T=0.8):
    '''
    evaluates model performance on the test set
    inputs:
    ------
    model_weights_path - a path to saved model weights to evaluate
    trest_generator - a keras data generator object, an output of
                      the create_data_generators function
    T - a threshold for confution matrix computation, any value larger than
        T would be regarded as positive detection (Y_pred > T)

    outputs:
    -------

    '''
    # create a model and load weights
    model = create_CNN_model()
    model.load_weights(model_weights_path)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # evaluate loss and accuracy values on a test batch
    X, Y = test_generator.__getitem__(0)
    loss, acc = model.evaluate(X, Y)

    # compute confusion matrix
    Y_pred = model.predict(X)
    plot_confusion_matrix(Y, Y_pred > T, ['Non Ad', 'Ad'])

    return loss, acc
