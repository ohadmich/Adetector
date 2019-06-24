import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
from sklearn.cluster import MiniBatchKMeans
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

"""This is a collection of utility functions used across the project"""

def create_model(n_features = 1690):
    '''Create a model obejct with an input of length n_features'''
    model = Sequential() # create a model instance

    #add model layers
    model.add(Dense(256, activation = 'relu', input_shape=(n_features,)))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    model.summary()
    return model

def create_CNN_model(n_mfcc = 13, n_timebins = 130, quiet = False):
    '''Create a model obejct with a 2D input of shape (n_mfcc, n_timebins,1)'''
    model = Sequential() # create a model instance

    #add model layers
    model.add(Conv2D(16, (3,3), strides=(1,1), activation = 'relu', padding='same', 
                     input_shape=(n_mfcc, n_timebins, 1)))
    model.add(Conv2D(16, (3,3), strides=(1,1), activation = 'relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    if not quiet:
        model.summary()
    
    return model

def audio2features_vectors(file_path, clip_duration = 3, sr = 22050, n_mfcc = 13, CNN=False):
    '''Takes a path to an audio file, cuts it to clip_duration windows,
       extracts features, normalizes and outputs a np.array of shape (n_clips, n_features)
       CNN flag is used for reshaping data for inputing to a CNN'''
    features_vec = []
    # load audio
    audio = librosa.core.load(file_path, sr = sr, duration = 20*60)[0]
    audio_length = len(audio)/sr # in sec
    n_clips = int(np.floor(audio_length/clip_duration)) # full clips in audio
    # cut audio to clips and extract features
    for i in range(n_clips):
        clip = audio[i*clip_duration*sr:(i+1)*clip_duration*sr]
        features = librosa.feature.mfcc(clip, sr=sr, n_mfcc=n_mfcc, dct_type=2)
        if CNN:
            features_vec.append(features)
        else:
            features_vec.append(features.flatten())
    
    X = np.array(features_vec)
    mu = np.mean(X, axis=0) 
    std = np.std(X, axis=0)
    X = (X-mu)/std
    
    if CNN:
        X = X.reshape((X.shape[0],X.shape[1], X.shape[2], 1))
        
    return X

def listen_to(file_path, start_time, end_time, sr=22050):
    '''Plays a part of audio file file_path, that starts at start_time
       and ends at end_time where start and end are given in minutes'''
    start_time_sec = start_time*60
    end_time_sec = end_time*60
    duration = end_time_sec - start_time_sec
    audio = librosa.core.load(file_path, sr = sr, offset = start_time_sec, duration = duration)[0]
    return ipd.Audio(audio, rate = sr)

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax