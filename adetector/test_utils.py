import unittest
import numpy as np

import utils

class TestUtils(unittest.TestCase):

    def test_moving_average_shape(self):
        n = 5
        a = np.arange(10)
        avg = utils.moving_average(a, n)
        self.assertAlmostEqual(avg.shape[0],a.shape[0])

if __name__ == '__main__':
    unittest.main()
