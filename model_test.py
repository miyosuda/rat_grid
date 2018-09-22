# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest

from model import Model


class ModelTest(unittest.TestCase):
    def test_init(self):
        model = Model(256, 12)
        self.assertEqual(model.g.get_shape()[1], 512)
        

if __name__ == '__main__':
    unittest.main()
