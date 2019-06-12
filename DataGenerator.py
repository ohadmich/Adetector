import os
import numpy as np
import librosa
import keras

class DataGenerator(keras.utils.Sequence):
    """Generate data for unsupervised learning"""
    def __init__(self, filepath_list, batch_size=10, sample_duration = 3, dataset='train', shuffle=True):
        """
        * filepath_list: a list of paths for audio files
        * batch_size: batch sample of each iteration
        * sample_duration: standard sample duration in the data set
        * dataset: use to label the dataset of the current generator
        * shuffle: whether or not shuffle the dataframe before itering over it. Default True!
        """
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.files = filepath_list
        self.n_files = len(self.files)
        self.sr = 22050 # audio sampling rate
        self.n_mfcc = 13 # number of frequency coefficients to use
        self.d = sample_duration
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
        np.random.shuffle(clip_list) # randomize clips (many of them come from the same file)
        for clip in clip_list:
            features = librosa.feature.mfcc(clip, sr=self.sr, n_mfcc=self.n_mfcc, dct_type=2)
            X.append(features.flatten())
            
        return np.vstack(X)
    
    def load_clips(self, filepath_list):
        '''Loads files in filepath_list, cuts them to clips of length
           d and returns a list of all the clips'''
        clip_list = []
        # load all files in filepath_list
        for f in filepath_list:
            try:
                audio = librosa.core.load(f, sr = self.sr)[0]
            except:
                self.err_files.append(f)
                continue
            # cut audio file to a self.d long clips
            audio_length = len(audio)/self.sr # in sec
            n_clips = int(np.floor(audio_length/self.d)) # full clips in audio
            for i in range(n_clips):
                clip_list.append(audio[i*self.d*self.sr:(i+1)*self.d*self.sr])          

        return clip_list
    
    def on_epoch_end(self):
            """Update indexes after each epoch"""
            if self.shuffle:
                np.random.shuffle(self.files)
                
class DataGenerator_Sup(keras.utils.Sequence):
    """Generate data for supervised learning"""
    def __init__(self, pos_files, neg_files = None, batch_size=10, sample_duration = 3, dataset='train', shuffle=True):
        """
        * pos_files: a list of paths for positive audio files (Ads)
        * neg_files: a list of paths for negative audio files (non Ads)
        * batch_size: batch sample of each iteration
        * sample_duration: standard sample duration in the data set
        * dataset: use to label the dataset of the current generator
        * shuffle: whether or not shuffle the dataframe before itering over it. Default True!
        """
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.pos_files = pos_files
        self.neg_files = neg_files
        self.n_files = len(self.pos_files) + len(self.neg_files)
        self.sr = 22050 # audio sampling rate
        self.n_mfcc = 13 # number of frequency coefficients to use
        self.d = sample_duration
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
        batch_files = self.pos_files[start_index:end_index]

        X = self.__data_generation(batch_files)

        return X

    def __data_generation(self, batch_files):
        """Generate a data batch"""
        X = []
        clip_list = self.load_clips(batch_files)
        np.random.shuffle(clip_list) # randomize clips (many of them come from the same file)
        for clip in clip_list:
            features = librosa.feature.mfcc(clip, sr=self.sr, n_mfcc=self.n_mfcc, dct_type=2)
            X.append(features.flatten())
            
        return np.vstack(X)
    
    def load_clips(self, filepath_list):
        '''Loads files in filepath_list, cuts them to clips of length
           d and returns a list of all the clips'''
        clip_list = []
        # load all files in filepath_list
        for f in filepath_list:
            try:
                audio = librosa.core.load(f, sr = self.sr)[0]
            except:
                self.err_files.append(f)
                continue
            # cut audio file to a self.d long clips
            audio_length = len(audio)/self.sr # in sec
            n_clips = int(np.floor(audio_length/self.d)) # full clips in audio
            for i in range(n_clips):
                clip_list.append(audio[i*self.d*self.sr:(i+1)*self.d*self.sr])          

        return clip_list
    
    def on_epoch_end(self):
            """Update indexes after each epoch"""
            if self.shuffle:
                np.random.shuffle(self.pos_files)