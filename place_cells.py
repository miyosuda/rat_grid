# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class PlaceCells(object):
    def __init__(self, cell_size=256, pos_range_min=-4.5, pos_range_max=4.5):
        self.cell_size = cell_size
        std = 0.04
        self.sigma_sq = std * std
        # Means of the gaussian
        self.us = np.random.rand(cell_size, 2) * (pos_range_max - pos_range_min) + pos_range_min

    def get_activation(self, pos):
        """
        Arguments:
          pos: Float Tuple(2)
        """
        d = self.us - pos
        norm2 = np.linalg.norm(d, ord=2, axis=1)
        cs = np.exp( -norm2 / (2.0 * self.sigma_sq) )
        return cs / np.sum(cs)

    def get_nearest_cell_pos(self, activation):
        index = np.argmax(activation)
        return self.us[index]
