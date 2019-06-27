""" This is a configuration file """

"""Package paths - Please update before installation!"""
# a path to a folder in which trained weights are saved
WEIGHTS_FOLDER = '/home/ohadmich/Documents/Github/Ad_Detector/adetector/model_weights'
# a path to a folder in which data for unitests is stored
TEST_DATA_FOLDER = '/home/ohadmich/Documents/Github/Ad_Detector/Data'
# a path to a folder which conatains positive examples
AD_FOLDER = '/home/ohadmich/Documents/Github/Data/audio_ads'
# a path to a folder which conatains negative music examples
MUSIC_FOLDER = '/home/ohadmich/Documents/Github/Data/Music'
# a path to a folder which conatains negative music examples
PODCAST_FOLDER = '/home/ohadmich/Documents/Github/Data/podcasts'

"""Model hyperparameters"""
SAMPLING_RATE = 22050 # the sampling rate of the audio file
CLIP_DURATION = 3 # length in seconds of the input clip to the classification model
N_MFCC = 13 # number of mfc coefficients used when extracting features
N_TIMEBINS = 130 # number of timebins in the  mfcc feature matrix of a 3s long clip.
                 # the shape of the mfcc feature matrix is (N_MFCC,N_TIMEBINS)
N_FEATURES = 1690 # used for NN model, where the input shape is (N_FEATURES,)

"""Data parameters"""
AD_FILE_DURATION = 30/60.0 # an average duration of an ad file in minutes
MUSIC_FILE_DURATION = 30/60.0 # music file duration in minutes
PODCAST_FILE_DURATION = 12/60.0 # podcast file duration in minutes