import unittest
import numpy as np

import core

class TestCore(unittest.TestCase):
    
    def setUp(self):
        self.audio_path = '../Data/sample_audio.wav'
        self.audio_length = 26 # sample_audio.wav's length in seconds
        self.d = 3 # default clip_duration in seconds
        self.n_mfcc = 13 # default number of mfc coeficients
        self.n_timebins = 130 # default timebins in each clip
        self.X_sample = np.load('../Data/X_sample.npy') # a sample X array for testing
        self.prob_over_time_sample = np.array([0.5,0.2,0.8,0.92,0.96,0.99,0.78,0.6,0.2])

    def test_audio2features_output_shape(self):
        X = core.audio2features(self.audio_path)
        self.assertAlmostEqual(X.shape[0], int(self.audio_length/self.d))
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

    def test_get_timestamps_output_shape(self):
        timestamps, probs = core.get_timestamps(self.prob_over_time_sample, n=1)
        self.assertAlmostEqual(timestamps.shape[0], 1)
        self.assertAlmostEqual(timestamps.shape[1], 2)
        self.assertAlmostEqual(probs.shape[0], 1)
        self.assertAlmostEqual(probs.shape[1], 1)

    def test_get_timestamps_values(self):
        timestamps, probs = core.get_timestamps(self.prob_over_time_sample, n=1)
        self.assertAlmostEqual(timestamps[0][0], 3*self.d/60)
        self.assertAlmostEqual(timestamps[0][1], 6*self.d/60)
        self.assertAlmostEqual(probs[0][0], 2.87/self.d)
    
    def test_get_timestamps_boundary_conditions(self):
        left_signal = np.array([0.98,0.95,0.92,0.8,0.5,0.6,0.75,0.5,0.2])
        right_signal = np.array([0.7,0.5,0.2,0.8,0.92,0.96,0.98,0.95,0.92])
        left_detection, left_probs = core.get_timestamps(left_signal, n=1)
        right_detection, right_probs = core.get_timestamps(right_signal, n=1)
        self.assertTrue(np.allclose(left_detection, np.array([[0, 3*self.d/60]])))
        self.assertTrue(np.allclose(right_detection, np.array([[4*self.d/60, 9*self.d/60]])))
        self.assertAlmostEqual(left_probs[0][0], (0.98+0.95+0.92)/3)
        self.assertAlmostEqual(right_probs[0][0], (0.92+0.96+0.98+0.95+0.92)/5)
        

if __name__ == '__main__':
    unittest.main()
