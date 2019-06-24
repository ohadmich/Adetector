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
        self.assertAlmostEqual(np.allclose(mu, zeros_array), True)
        self.assertAlmostEqual(np.allclose(std, ones_array), True)


if __name__ == '__main__':
    unittest.main()
