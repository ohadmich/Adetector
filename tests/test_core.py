import os
import unittest
import numpy as np

import adetector as adt
from adetector.config import TEST_DATA_FOLDER


class TestCore(unittest.TestCase):

    def setUp(self):
        self.audio_path = os.path.join(TEST_DATA_FOLDER, 'sample_audio.wav')
        self.audio_length = 26  # sample_audio.wav's length in seconds
        self.d = 3  # default clip_duration in seconds
        self.n_mfcc = 13  # default number of mfc coeficients
        self.n_timebins = 130  # default timebins in each clip
        # a sample X array for testing
        self.X_sample = np.load(os.path.join(TEST_DATA_FOLDER, 'X_sample.npy'))
        self.prob_over_time_sample =\
             np.array([0.5, 0.2, 0.8, 0.92, 0.96, 0.99, 0.78, 0.6, 0.2])
        self.timestamps_sample =\
             np.array([[0., 0.15], [0.3, 0.4]])  # timestamps for sample audio
        # ad probabilities of sample audio
        self.probs_sample = np.array([[0.21392338], [0.9058739]])

    def test_audio2features_output_shape(self):
        X = adt.core.audio2features(self.audio_path)
        self.assertAlmostEqual(X.shape[0], int(self.audio_length/self.d))
        self.assertAlmostEqual(X.shape[1], self.n_mfcc)
        self.assertAlmostEqual(X.shape[2], self.n_timebins)
        self.assertAlmostEqual(X.shape[3], 1)  # X has only one channel
    
    def test_audio2features_output_normalization(self):
        X = adt.core.audio2features(self.audio_path)
        mu = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        zeros_array = np.zeros((self.n_mfcc, self.n_timebins, 1))
        ones_array = np.ones((self.n_mfcc, self.n_timebins, 1))
        self.assertTrue(np.allclose(mu, zeros_array))
        self.assertTrue(np.allclose(std, ones_array))

    def test_Ad_vs_music_classifier_output_shape(self):
        prob_over_time = adt.core.Ad_vs_music_classifier(self.X_sample)
        self.assertAlmostEqual(prob_over_time.shape[0], self.X_sample.shape[0])
        self.assertAlmostEqual(prob_over_time.shape[1], 1)

    def test_Ad_vs_music_classifier_output_value_range(self):
        prob_over_time = adt.core.Ad_vs_music_classifier(self.X_sample)
        self.assertTrue(np.max(prob_over_time) <= 1)
        self.assertTrue(np.min(prob_over_time) >= 0)

    def test_get_timestamps_output_shape(self):
        timestamps, probs = adt.core.get_timestamps(self.prob_over_time_sample,
                                                    n=1)
        self.assertAlmostEqual(timestamps.shape[0], 1)
        self.assertAlmostEqual(timestamps.shape[1], 2)
        self.assertAlmostEqual(probs.shape[0], 1)
        self.assertAlmostEqual(probs.shape[1], 1)

    def test_get_timestamps_values(self):
        timestamps, probs = adt.core.get_timestamps(self.prob_over_time_sample,
                                                    n=1)
        self.assertAlmostEqual(timestamps[0][0], 3*self.d/60)
        self.assertAlmostEqual(timestamps[0][1], 6*self.d/60)
        self.assertAlmostEqual(probs[0][0], 2.87/self.d)

    def test_get_timestamps_boundary_conditions(self):
        left_signal = np.array([0.98, 0.95, 0.92, 0.8, 0.5, 0.6, 0.75, 0.5, 0.2])
        right_signal = np.array([0.7, 0.5, 0.2, 0.8, 0.92, 0.96, 0.98, 0.95, 0.92])
        left_detection, left_probs = adt.core.get_timestamps(left_signal, n=1)
        right_detection, right_probs = adt.core.get_timestamps(right_signal,
                                                               n=1)
        self.assertTrue(np.allclose(left_detection,
                                    np.array([[0, 3*self.d/60]])))
        self.assertTrue(np.allclose(right_detection,
                                    np.array([[4*self.d/60, 9*self.d/60]])))
        self.assertAlmostEqual(left_probs[0][0], (0.98+0.95+0.92)/3)
        self.assertAlmostEqual(right_probs[0][0], (0.92+0.96+0.98+0.95+0.92)/5)
    
    def test_extract_timeframe_indices_outputs(self):
        t = np.arange(10)*2
        S_idx, E_idx = adt.core.extract_timeframe_indices(t, 10, 18)
        self.assertAlmostEqual(S_idx, 6)
        self.assertAlmostEqual(E_idx, 10)

    def test_Ad_vs_speech_classifier_output_shape(self):
        probs = adt.core.Ad_vs_speech_classifier(self.X_sample,
                                                 self.timestamps_sample,
                                                 np.array([[0.99940175],[0.9953462]]))
        self.assertAlmostEqual(probs.shape[0], self.probs_sample.shape[0])
        self.assertAlmostEqual(probs.shape[1], self.probs_sample.shape[1])

    def test_Ad_vs_speech_classifier_output_value(self):
        probs = adt.core.Ad_vs_speech_classifier(self.X_sample,
                                                 self.timestamps_sample,
                                                 np.array([[0.99940175],[0.9953462]]))
        self.assertAlmostEqual(probs[0][0], self.probs_sample[0][0])
        self.assertAlmostEqual(probs[1][0], self.probs_sample[1][0])     

if __name__ == '__main__':
    unittest.main()
