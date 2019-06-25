import unittest
import numpy as np

import utils

class TestUtils(unittest.TestCase):

    def test_moving_average_shape(self):
        n = 5
        a = np.arange(10)
        avg = utils.moving_average(a, n)
        self.assertAlmostEqual(avg.shape[0],a.shape[0])
    
    def test_moving_average_output_value(self):
        n = 3
        a = np.arange(10)
        avg = utils.moving_average(a, n)
        result = np.array([1.0/3,1,2,3,4,5,6,7,8,26/3])
        self.assertTrue(np.allclose(avg, result))

if __name__ == '__main__':
    unittest.main()
