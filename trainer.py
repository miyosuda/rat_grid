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
        out = self.data_manager.get_train_batch(self.flags.batch_size,
                                                self.flags.sequence_length)
        inputs_batch, place_outputs_batch, hd_outputs_batch, place_init_batch, hd_init_batch = \
            out
        _, loss = sess.run([self.train_op, self.model.total_loss],
                           feed_dict = {
                               self.model.inputs : inputs_batch,
                               self.model.place_outputs :place_outputs_batch, 
                               self.model.hd_outputs : hd_outputs_batch, 
                               self.model.place_init : place_init_batch,
                               self.model.hd_init : hd_init_batch,
                               self.model.keep_prob : 0.5
                           })
        print(loss)
