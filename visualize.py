# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

from model import Model
from data_manager import DataManager
from hd_cells import HDCells
from place_cells import PlaceCells

def load_checkpoints(sess):
    saver = tf.train.Saver(max_to_keep=2)
    save_dir = "./saved"
    checkpoint_dir = save_dir + "/checkpoints"
    
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)

np.random.seed(1)
    
data_manager = DataManager()

place_cells = PlaceCells()
hd_cells = HDCells()

data_manager.prepare(place_cells, hd_cells)

model = Model(place_cell_size=place_cells.cell_size,
                          hd_cell_size=hd_cells.cell_size,
                          sequence_length=100)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Load checkpoints
load_checkpoints(sess)

batch_size = 10
sequence_length = 100
resolution = 20
maze_extents = 4.5

activations = np.zeros([512, resolution, resolution], dtype=np.float32) # (512, 32, 32)
counts  = np.zeros([resolution, resolution], dtype=np.int32)        # (32, 32)

index_size = data_manager.get_confirm_index_size(batch_size, sequence_length)

for index in range(index_size):
    out = data_manager.get_confirm_batch(batch_size, sequence_length, index)
    inputs_batch, place_init_batch, hd_init_batch, place_pos_batch = out
            
    place_pos_batch = np.reshape(place_pos_batch, [-1, 2])
    # (1000, 2)
    
    g = sess.run(
            model.g,
            feed_dict = {
                model.inputs : inputs_batch,
                model.place_init : place_init_batch,
                model.hd_init : hd_init_batch,
                model.keep_prob : 1.0
            })
    
        
    for i in range(batch_size * sequence_length):
        pos_x = place_pos_batch[i,0]
        pos_z = place_pos_batch[i,1]
        x = (pos_x + maze_extents) / (maze_extents * 2) * resolution
        z = (pos_z + maze_extents) / (maze_extents * 2) * resolution
        counts[int(x), int(z)] += 1
        activations[:, int(x), int(z)] += np.abs(g[i, :])

for x in range(resolution):
    for y in range(resolution):
        if counts[x, y] > 0:
            activations[:, x, y] /= counts[x, y]

activations = (activations - np.min(activations)) / (np.max(activations) - np.min(activations))

hidden_size = 512

plt.figure(figsize=(15, 25))
for i in range(hidden_size):
    plt.subplot(hidden_size//16, 16, 1 + i)
    plt.imshow(activations[i,:,:], interpolation="gaussian", cmap="jet")
    plt.axis('off')
plt.savefig("grids.png")
plt.close()
