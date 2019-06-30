"""
This file contains all the functions needed for applying
the detection algorithm on an audio file.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.callbacks import ModelCheckpoint

from . import utils
from .config import (WEIGHTS_FOLDER, SAMPLING_RATE,
                     CLIP_DURATION, N_MFCC, N_TIMEBINS)
from .models import create_CNN_model


def find_ads(X, T=0.85, n=10, show=False):
    '''Returns timestamps and probabilities for all ads found in feature array X
       inputs:
       ------
       X - feature array prepared out of an audio file using audio2features
       T - a probability threshold for detection
       n - moving average window size, used for smoothing before detection
       show - when set to be true, shows a plot of ad probability over time

       outputs:
       -------
       timestamps - a 2D array containing the detected timestamps
                    shape: (n_detections, 2)
       probs - a vector of ad probabilities for each detection
               shape: (n_detection,)

    '''
    prob_over_time = Ad_vs_music_classifier(X)
    timestamps, probs = get_timestamps(prob_over_time, T, n, show)
    probs = Ad_vs_speech_classifier(X, timestamps, probs)

    return timestamps, probs


def audio2features(file_path, offset=0.0, max_duration=20, CNN=True):
    ''' Prepares an array of features to which are used as an input to a model
        for prediction. The entire file is devided into short clips, features
        are extracted from each clip using mfccs and then normalized
        inputs:
        ------
        file_path - a path to a radio stream audio file
        offset - load audio with offset from start time, shoud be in minutes!
        max_duration - maximal duration being loaded into memory,
                       should be given in minutes!
        CNN - true for inputing features to a CNN model

        outputs:
        -------
        X - an array of normalized features, in shape:
            *(n_clips, n_features,) for NN (when CNN is False)
            *(n_clips, n_mfcc, n_timebins, 1) for CNN (when CNN is True)
    '''
    sr = SAMPLING_RATE
    d = CLIP_DURATION
    features_vec = []
    # load audio
    audio = librosa.core.load(file_path, sr=sr, offset=offset*60,
                              duration=max_duration*60)[0]
    audio_length = len(audio)/sr  # in sec
    n_clips = int(np.floor(audio_length/d))  # full clips in audio
    # cut audio to clips and extract features
    for i in range(n_clips):
        clip = audio[i*d*sr:(i+1)*d*sr]
        features = librosa.feature.mfcc(clip, sr=sr, n_mfcc=N_MFCC, dct_type=2)
        if CNN:
            features_vec.append(features)
        else:
            features_vec.append(features.flatten())

    X = np.array(features_vec)
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X-mu)/std

    if CNN:
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))

    return X


def Ad_vs_music_classifier(X):
    '''Returns a vector of Ad probabilities for each point in time
       based on a model trained to differentiate between ads and music
       inputs:
       ------
       X - an array of features in shape: (n_clips, n_mfcc, n_timebins, 1)
       outputs:
       -------
       prob_over_time - a vector of probability in each clip, shape (n_clips,)
    '''
    # choose model's weights
    weights_filename = \
        'weights_LeNet5ish_1000_only_music_and_ads_10epochs.hdf5'
    weights_path = os.path.join(WEIGHTS_FOLDER, weights_filename)
    # create a model for evaluation
    model = create_CNN_model(quiet=True)
    # load weights
    model.load_weights(weights_path)
    # compile
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # predict probabilities using the models
    prob_over_time = model.predict(X)

    return prob_over_time


def get_timestamps(prob_over_time, T=0.85, n=10, show=False):
    '''Takes a probility over time and extracts timestamps of detected ads
       inputs:
       ------
       prob_over_time - a vector of ad probabilities for every clip.
       T - a probability threshold for detection
       n - moving average window size, used for smoothing before detection
       show - when set to be true, shows a plot of ad probability over time
       outputs:
       -------
       timestamps - a 2D array of with the detected timestamps
                    shape: (n_detections, 2)
       probs - a vector of ad probabilities for each detection, calculated only
               based on music vs. ad classifier
               (gives high probability for speech), shape: (n_detection,)
    '''
    d = CLIP_DURATION
    # smoothing and creating a time axis
    prob_over_time_smooth = utils.moving_average(prob_over_time.flatten(), n)
    t = np.arange(1, len(prob_over_time)+1)*d/60.0  # create time axis

    # thresholding for detection and extracting start and end times
    detection_domains = (prob_over_time_smooth > T).astype(int)
    transitions = np.diff(detection_domains)  # get transitions

    start_times = t[:-1][transitions == 1]  # ad starts when prob. goes 0 -> 1
    end_times = t[:-1][transitions == -1]  # ad ends when prob. goes 1 -> 0

    # if no signal was detected, return 0
    if start_times.size == 0 and end_times.size == 0:
        return 0, 0
    # if only start time was detected, add the last time frame as end time
    elif end_times.size == 0:
        end_times = np.array([t[-1]])
    # if only end time was detected, add 0 as a start time
    elif start_times.size == 0:
        start_times = np.array([0])
    # otherwise check boundary conditions for the detection
    else:
        # if the first event is an end - add a start event at time 0
        if np.min(end_times) < np.min(start_times):
            start_times = np.hstack([0, start_times])
        # if the last event is a start - add an end event at the end
        if np.max(end_times) < np.max(start_times):
            end_times = np.hstack([end_times, t[-1]])

    timestamps = np.vstack([start_times, end_times]).transpose()

    # compute probabilities based on music vs ad classifier
    music_vs_ads_probs = []
    for i in range(timestamps.shape[0]):
        S_idx, E_idx = extract_timeframe_indices(t, timestamps[i][0],
                                                 timestamps[i][1])
        prob = np.mean(prob_over_time[S_idx:E_idx])

        music_vs_ads_probs.append(prob)

    # if show is set to true, plot ad probability over time and threshold
    if show:
        plt.figure(figsize=(8, 5))
        plt.subplot(3, 1, 1)
        plt.plot(t, prob_over_time, 'g')
        plt.ylabel('Ad prob.')

        plt.subplot(3, 1, 2)
        threshold = np.ones(len(t))*T
        plt.plot(t, prob_over_time_smooth, 'r')
        plt.plot(t, threshold)
        plt.ylabel('Ad prob.')
        plt.legend(['smooth ad probability', 'threshold'])

        plt.subplot(3, 1, 3)
        plt.plot(t, prob_over_time_smooth > T, 'bo')
        plt.xlabel('Time (min)')
        plt.ylabel('Detection')
        plt.tight_layout()

        plt.show()

    return timestamps, np.vstack(music_vs_ads_probs)


def extract_timeframe_indices(t, start_time, end_time):
    '''Returns the indices of a timeframe starting at start_time
       and ending and end_time
       inputs:
       ------
       t - a time vector in minutes, shape: (n_clips, 1)
       start_time - start of time frame in minutes
       end_time - end of time frame in minutes

       outputs:
       -------
       S_idx - start index of the time frame
       E_idx - end index of the time frame
    '''
    # handle left boundary case - no start detection
    if start_time == 0:
        S_idx = 0
    else:
        S_idx = np.argmin((t-start_time)**2) + 1  # start index
    E_idx = np.argmin((t-end_time)**2) + 1  # end index

    return S_idx, E_idx


def Ad_vs_speech_classifier(X, timestamps, probs):
    '''Returns ad probability vector after applying ad vs speech classification
       inputs:
       ------
       X - an array of features in shape: (n_clips, n_mfcc, n_timebins, 1)
       timestamps - a 2D array of with the detected timestamps
                    shape: (n_detections, 2)
       probs - a vector of ad probabilities for each detection, calculated only
               based on music vs. ad classifier
               (gives high probability for speech), shape: (n_detection,)
       outputs:
       -------
       probs - probability of vector of shape (len(timestamps),) containing
               the probability of each timestamp being an ad based on a model
               trained to differentiate between ads and podcasts
    '''
    d = CLIP_DURATION
    # choose model's weights
    weights_filename = \
        'weights_LeNet5ish_1000_only_podcasts_and_ads_6epochs.hdf5'
    weights_path = os.path.join(WEIGHTS_FOLDER, weights_filename)
    # recreate a model for evaluation
    speech_model = create_CNN_model(quiet=True)
    # load weights
    speech_model.load_weights(weights_path)
    # compile
    speech_model.compile(loss='binary_crossentropy',
                         optimizer='adam', metrics=['accuracy'])

    t = np.arange(1, X.shape[0]+1)*d/60.0  # create time axis
    for i in range(timestamps.shape[0]):
        S_idx, E_idx = extract_timeframe_indices(t, timestamps[i][0],
                                                 timestamps[i][1])
        X_p = X[S_idx:E_idx]
        Ad_prob = speech_model.predict(X_p)
        probs[i] = probs[i] * np.mean(Ad_prob)

    return probs
