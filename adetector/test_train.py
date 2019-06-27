import unittest
import numpy as np

import train
import config

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.n_ads = 2293 # number of ad files
        self.n_music = 1013 # number of music files
        self.n_podcasts = 300 # number of podcast files
        self.pos_files = np.load('../Data/pos_file_paths.npy')
        self.music_files = np.load('../Data/music_file_paths.npy')
        self.podcast_files = np.load('../Data/podcast_file_paths.npy')

    def test_list_data_output(self):
        a,m,p = train.list_data()
        self.assertAlmostEqual(len(a),self.n_ads)
        self.assertAlmostEqual(len(m),self.n_music)
        self.assertAlmostEqual(len(p),self.n_podcasts)
    
    def test_create_data_generators_length(self):
        trng, tstg = train.create_data_generators(self.pos_files, self.music_files)
        self.assertAlmostEqual(len(trng), 180)
        self.assertAlmostEqual(len(tstg), 20)

    def test_create_data_generators_output_shape(self):
        trng, tstg = train.create_data_generators(self.pos_files, self.music_files)
        X, Y = trng.__getitem__(0)
        self.assertTupleEqual(X.shape[1:], (config.N_MFCC, config.N_TIMEBINS, 1))
        self.assertAlmostEqual(Y.shape[1], 1)
        self.assertAlmostEqual(X.shape[0], Y.shape[0])
        X1, Y1 = tstg.__getitem__(0)
        self.assertTupleEqual(X1.shape[1:], (config.N_MFCC, config.N_TIMEBINS, 1))
        self.assertAlmostEqual(Y1.shape[1], 1)
        self.assertAlmostEqual(Y1.shape[0], X1.shape[0])

    def test_create_data_generators_length_podcast_case(self):
        trng, tstg = train.create_data_generators(self.pos_files, 
                                                  self.podcast_files, 
                                                  neg_type=True, data_minutes=100)
        self.assertAlmostEqual(len(trng), 31)
        self.assertAlmostEqual(len(tstg), 3)

    def test_create_data_generators_assertion_error_not_enough_files(self):
        self.assertRaises(AssertionError, lambda: train.create_data_generators(
                                                  self.pos_files, 
                                                  self.podcast_files, 
                                                  neg_type=True))

if __name__ == '__main__':
    unittest.main()
