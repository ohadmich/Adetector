import unittest
import numpy as np

import core

class TestCore(unittest.TestCase):
    
    def setUp(self):
        self.audio_path = '../Data/sample_audio.wav'
        self.audio_length = 27 # sample_audio.wav's length in seconds
        self.d = 3 # default clip_duration in seconds
        self.n_mfcc = 13 # default number of mfc coeficients
        self.n_timebins = 130 # default timebins in each clip
        self.X_sample = np.load('../Data/X_sample.npy') # a sample X array for testing

    def test_audio2features_output_shape(self):
        X = core.audio2features(self.audio_path)
        self.assertAlmostEqual(X.shape[0], self.audio_length/self.d)
        self.assertAlmostEqual(X.shape[1], self.n_mfcc)
        self.assertAlmostEqual(X.shape[2], self.n_timebins)
        self.assertAlmostEqual(X.shape[3],1) # X has only one channel
    
    def test_audio2features_output_normalization(self):
        X = core.audio2features(self.audio_path)
        mu = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        zeros_array = np.zeros((self.n_mfcc, self.n_timebins, 1))
        ones_array = np.ones((self.n_mfcc, self.n_timebins, 1))
        self.assertTrue(np.allclose(mu, zeros_array))
        self.assertTrue(np.allclose(std, ones_array))

    def test_Ad_vs_music_classifier_output_shape(self):
        prob_over_time = core.Ad_vs_music_classifier(self.X_sample)
        self.assertAlmostEqual(prob_over_time.shape[0],self.X_sample.shape[0])
        self.assertAlmostEqual(prob_over_time.shape[1],1)

    def test_Ad_vs_music_classifier_output_value_range(self):
        prob_over_time = core.Ad_vs_music_classifier(self.X_sample)
        self.assertTrue(np.max(prob_over_time)<=1)
        self.assertTrue(np.min(prob_over_time)>=0)

if __name__ == '__main__':
    unittest.main()
