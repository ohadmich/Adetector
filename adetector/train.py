import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import keras
from keras.callbacks import ModelCheckpoint

from models import create_CNN_model
from utils import plot_confusion_matrix
from DataGenerator import DataGenerator_Sup
import config

def list_data():
    '''
    lists all data present in the directories defined in config.py
    returns a list of paths to positive files, music files and podcast files 
    '''
    pos_files = [] # a list of paths to positive examples
    for r,d,f in os.walk(config.AD_FOLDER):
        for filename in f:
            if '.mp3' in filename:
                pos_files.append(os.path.join(r,filename))

    music_files = [] # a list of paths to negative music examples
    for r,d,f in os.walk(config.MUSIC_FOLDER):
        for filename in f:
            if '.mp3' or '.au' in filename:
                music_files.append(os.path.join(r,filename))

    podcast_files = [] # a list of paths to negative podcast examples
    for r,d,f in os.walk(config.PODCAST_FOLDER):
        for filename in f:
            if '.wav' in filename:
                podcast_files.append(os.path.join(r,filename))
    
    neg_files = music_files + podcast_files # a list of paths of all negative examples
    n_pos_files = len(pos_files) # number of positive files
    n_neg_files = len(neg_files) # number of negative files

    print('There are ' + str(n_pos_files) + ' positive examples')
    print('There are ' + str(len(music_files)) + ' music examples')
    print('There are ' + str(len(podcast_files)) + ' podcast examples')

    # compute the amount of data in minutes
    pos_minutes = round(config.AD_FILE_DURATION*n_pos_files,2)
    neg_minutes = round(config.MUSIC_FILE_DURATION*len(music_files) +
                        config.PODCAST_FILE_DURATION*len(podcast_files),2)
    print('--------------------------------')
    print('In total, ' + str(pos_minutes) + ' minutes of positive and ' 
          + str(neg_minutes) + ' minutes of negative')
    
    return pos_files, music_files, podcast_files
