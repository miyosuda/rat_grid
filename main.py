# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from model import Model
from trainer import Trainer
from data_manager import DataManager
from hd_cells import HDCells
from place_cells import PlaceCells


def get_options():
    tf.app.flags.DEFINE_string("save_dir", "saved", "checkpoints,log,options save directory")
    tf.app.flags.DEFINE_integer("batch_size", 10, "batch size")
    tf.app.flags.DEFINE_integer("sequence_length", 100, "sequence length")    
    tf.app.flags.DEFINE_integer("steps", 300000, "training steps")
    tf.app.flags.DEFINE_integer("save_interval", 5000, "saving interval")
    tf.app.flags.DEFINE_float("learning_rate", 1e-5, "learning rate")
    tf.app.flags.DEFINE_float("momemtum", 0.9, "momemtum")
    tf.app.flags.DEFINE_float("weight_decay", 1e-5, "weight decay")
    tf.app.flags.DEFINE_float("gradient_clipping", 1e-5, "gradient clipping")
    return tf.app.flags.FLAGS

flags = get_options()

def load_checkpoints(sess):
    saver = tf.train.Saver(max_to_keep=2)
    checkpoint_dir = flags.save_dir + "/checkpoints"
    
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        # checkpointからロード
        saver.restore(sess, checkpoint.model_checkpoint_path)
        # ファイル名から保存時のstep数を復元
        tokens = checkpoint.model_checkpoint_path.split("-")
        step = int(tokens[1])
        print("Loaded checkpoint: {0}, step={1}".format(checkpoint.model_checkpoint_path, step))
        return saver, step+1
    else:
        print("Could not find old checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        return saver, 0


def save_checkponts(sess, saver, global_step):
    checkpoint_dir = flags.save_dir + "/checkpoints"
    saver.save(sess, checkpoint_dir + '/' + 'checkpoint', global_step=global_step)
    print("Checkpoint saved")


def train(sess, trainer, saver, summary_writer, start_step):
    for i in range(start_step, flags.steps):
        # 学習
        trainer.train(sess, summary_writer, step=i)
        
        if i % flags.save_interval == flags.save_interval-1:
            # 保存
            save_checkponts(sess, saver, i)
        

def main(argv):
    np.random.seed(1)
    
    if not os.path.exists(flags.save_dir):
        os.mkdir(flags.save_dir)

    data_manager = DataManager()

    place_cells = PlaceCells()
    hd_cells = HDCells()

    data_manager.prepare(place_cells, hd_cells)

    model = Model(place_cell_size=place_cells.cell_size,
                  hd_cell_size=hd_cells.cell_size,
                  sequence_length=flags.sequence_length)
    
    trainer = Trainer(data_manager, model, flags)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # For Tensorboard log
    log_dir = flags.save_dir + "/log"
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    # Load checkpoints
    saver, start_step = load_checkpoints(sess)

    # Train
    train(sess, trainer, saver, summary_writer, start_step)

    

if __name__ == '__main__':
    tf.app.run()
