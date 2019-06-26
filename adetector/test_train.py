import unittest
import numpy as np

import train

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.n_ads = 2293 # number of ad files
        self.n_music = 1013 # number of music files
        self.n_podcasts = 300 # number of podcast files

    def test_list_data_output(self):
        a,m,p = train.list_data()
        self.assertAlmostEqual(len(a),self.n_ads)
        self.assertAlmostEqual(len(m),self.n_music)
        self.assertAlmostEqual(len(p),self.n_podcasts)

if __name__ == '__main__':
    unittest.main()
