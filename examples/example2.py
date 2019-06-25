import sys
sys.path.insert(0, '../adetector')

import core
import utils

'''load audio file and convert it to an array with shape 
   (n_timeframes, n_mfcc, n_timebins, 1)
   where n_timeframes = audio_length/d 
'''
radio_stream_path =  '../Data/Z100 Stream Recording.mp3'
X = core.audio2features(radio_stream_path)
timestamps, probs = core.find_ads(X, T=0.85, n=10, show=True)

print("Ads were detected at the following timestamps:")
print(timestamps)
print("Ad probabilities for these timestamps are:")
print(probs)

utils.listen_to(radio_stream_path, *timestamps[1])