import os
import numpy as np
import librosa
import keras

from . import config


class DataGenerator(keras.utils.Sequence):
    """Generate data for unsupervised learning"""
    def __init__(self, filepath_list, batch_size=10,
                 dataset='train', shuffle=True):
        """
        * filepath_list: a list of paths for audio files
        * batch_size: batch sample of each iteration
        * dataset: use to label the dataset of the current generator
        * shuffle: whether or not shuffle the dataframe before itering over it.
                   Default True!
        """
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.files = filepath_list
        self.n_files = len(self.files)
        self.sr = config.SAMPLING_RATE  # audio sampling rate
        self.n_mfcc = config.N_MFCC  # number of frequency coefficients to use
        self.d = config.CLIP_DURATION  # length in seconds of the model's input
        self.on_epoch_end()
        self.err_files = []

    def __len__(self):
        """Denotes the number of batches per epoch"""
        num_batches = int(np.floor(self.n_files/self.batch_size))

        return num_batches

    def __getitem__(self, index):
        """Generate one batch of data"""
        start_index = index * self.batch_size
        end_index = (index+1) * self.batch_size
        if end_index > self.n_files:
            end_index = None
        batch_files = self.files[start_index:end_index]

        X = self.__data_generation(batch_files)

        return X

    def __data_generation(self, batch_files):
        """Generate a data batch"""
        X = []
        clip_list = self.load_clips(batch_files)
        # randomize clips (many of them come from the same file)
        np.random.shuffle(clip_list)
        for clip in clip_list:
            features = librosa.feature.mfcc(clip, sr=self.sr,
                                            n_mfcc=self.n_mfcc, dct_type=2)
            X.append(features.flatten())

        return np.vstack(X)

    def load_clips(self, filepath_list):
        '''Loads files in filepath_list, cuts them to clips of length
           d and returns a list of all the clips'''
        clip_list = []
        # load all files in filepath_list
        for f in filepath_list:
            try:
                audio = librosa.core.load(f, sr=self.sr)[0]
            except:
                self.err_files.append(f)
                continue
            # cut audio file to a self.d long clips
            audio_length = len(audio)/self.sr  # in sec
            n_clips = int(np.floor(audio_length/self.d))  # full clips in audio
            for i in range(n_clips):
                clip_list.append(audio[i*self.d*self.sr:(i+1)*self.d*self.sr])          

        return clip_list

    def on_epoch_end(self):
            """Update indexes after each epoch"""
            if self.shuffle:
                np.random.shuffle(self.files)


class DataGenerator_Sup(keras.utils.Sequence):
    """Generate data for supervised learning"""
    def __init__(self, files, batch_size=10, dataset='train',
                 shuffle=True, CNN=False):
        """
        * files: a list of data files in which each element contains
                 a list of the form [path, label] where label is
                 denoted 1 for Ads and 0 for Non Ads
        * batch_size: batch sample of each iteration
        * dataset: use to label the dataset of the current generator
        * shuffle: whether or not shuffle the dataframe before itering over it.
                   Default True!
        * CNN: outputs X in shape (n_samples, n_mfcc, n_timebins)
               when True for training CNN models
        """
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.CNN = CNN
        self.files = files
        self.n_files = len(files)
        self.sr = config.SAMPLING_RATE  # audio sampling rate
        self.n_mfcc = config.N_MFCC  # number of frequency coefficients to use
        self.d = config.CLIP_DURATION  # standard sample duration in the data
        self.err_files = []
        self.mu = 0
        self.std = 0

        # shuffle files
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        num_batches = int(self.n_files/self.batch_size)

        return num_batches

    def __getitem__(self, index):
        """Generate one batch of data"""
        start_index = index * self.batch_size
        end_index = (index+1) * self.batch_size
        if end_index > self.n_files:
            end_index = None

        batch_files = []
        labels = []
        for i in range(start_index, end_index):
            batch_files.append(self.files[i][0])
            labels.append(self.files[i][1])

        X, Y = self.__data_generation(batch_files, labels)
        '''Normalization'''
        self.mu = np.mean(X, axis=0) 
        self.std = np.std(X, axis=0)
        X = (X-self.mu)/self.std

        # reshape X for CNN models as an image with one channel
        if self.CNN:
            X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))

        return X, Y

    def __data_generation(self, batch_files, labels):
        """Generate a data batch"""
        X = []
        Y = []
        clip_list = self.load_clips(batch_files, labels)
        # randomize clips (many of them come from the same file)
        np.random.shuffle(clip_list)
        for clip in clip_list:
            features = librosa.feature.mfcc(clip[0], sr=self.sr,
                                            n_mfcc=self.n_mfcc, dct_type=2)
            Y.append(clip[1])

            if self.CNN:
                X.append(features)
            else:
                X.append(features.flatten())

        return np.array(X), np.vstack(Y).astype(int)

    def load_clips(self, filepath_list, labels):
        '''Loads files in filepath_list, cuts them to clips of length
           d and returns a list of all the clips'''
        clip_list = []
        # load all files in filepath_list
        for i, f in enumerate(filepath_list):
            try:
                audio = librosa.core.load(f, sr=self.sr)[0]
                label = labels[i]
            except:
                self.err_files.append(f)
                continue
            # cut audio file to a self.d long clips
            audio_length = len(audio)/self.sr  # in sec
            n_clips = int(np.floor(audio_length/self.d))  # full clips in audio
            for i in range(n_clips):
                clip_list.append([audio[i*self.d*self.sr:(i+1)*self.d*self.sr], label])

        return clip_list

    def on_epoch_end(self):
            """Update indexes after each epoch"""
            if self.shuffle:
                np.random.shuffle(self.files)
