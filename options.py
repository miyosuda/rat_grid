# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def get_options():
    tf.app.flags.DEFINE_string("save_dir", "saved", "checkpoints,log,options save directory")
    tf.app.flags.DEFINE_integer("batch_size", 10, "batch size")
    tf.app.flags.DEFINE_integer("sequence_length", 100, "sequence length")    
    tf.app.flags.DEFINE_integer("steps", 300000, "training steps")
    tf.app.flags.DEFINE_integer("save_interval", 5000, "saving interval")
    tf.app.flags.DEFINE_float("learning_rate", 1e-5, "learning rate")
    tf.app.flags.DEFINE_float("momentum", 0.9, "momentum")
    tf.app.flags.DEFINE_float("l2_reg", 1e-5, "weight decay")
    tf.app.flags.DEFINE_float("gradient_clipping", 1e-5, "gradient clipping")
    return tf.app.flags.FLAGS

