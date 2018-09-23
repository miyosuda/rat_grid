# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class Trainer(object):
    def __init__(self, data_manager, model, flags):
        self.data_manager = data_manager
        self.model = model
        self.flags = flags

        self._prepare_optimizer()
      
    def _prepare_optimizer(self):
        # TODO: RMSProp
        self.train_op = tf.train.AdamOptimizer(self.flags.learning_rate).minimize(
            self.model.total_loss)
        # TODO: gradient clipping
        # TODO: weight decay

    def train(self, sess, summary_writer, step):
        # TODO:
        pass
