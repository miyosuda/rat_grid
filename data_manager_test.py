# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import unittest

from data_manager import DataManager


class DataManagerTest(unittest.TestCase):
    def setUp(self):
        self.data_manager = DataManager()
    
    def test_value_range(self):
        # Pos range is -4.5 ~ 4.5 (実際は-4.03~4.03あたり)
        self.assertLessEqual(   np.max(self.data_manager.pos_xs),  4.5)
        self.assertGreaterEqual(np.min(self.data_manager.pos_xs), -4.5)
        
        # Angle range is -pi ~ pi
        self.assertLessEqual(   np.max(self.data_manager.angles),  np.pi)
        self.assertGreaterEqual(np.min(self.data_manager.angles), -np.pi)
        
    def test_data_shape(self):
        # Check data shape
        self.assertEqual(self.data_manager.velocity_xs.shape,         (49999,))
        self.assertEqual(self.data_manager.velocity_zs.shape,         (49999,))
        self.assertEqual(self.data_manager.anglular_velocities.shape, (49999,))
        
        self.assertEqual(self.data_manager.angles.shape,              (49999,))
        self.assertEqual(self.data_manager.pos_xs.shape,              (49999,))
        self.assertEqual(self.data_manager.pos_zs.shape,              (49999,))
        
            
if __name__ == '__main__':
    unittest.main()
