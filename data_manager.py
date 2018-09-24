# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle

# データ内の1エピソードのサイズ
EPISODE_LENGTH = 400


class DataManager(object):
    POS_RANGE_MAX = 4.5
    POS_RANGE_MIN = -4.5
    
    def __init__(self):
        data = pickle.load(open("data/data.p", "rb"))
        input_x = data["x"] # (49999, 7)
        # angle(-1~1), vel_x, vel_z, angular_vel, velocity_magnigude, heading?, velocity?
        # vel_x, vel_zからvelocity_magnitudeに等しいものが出るのは確認できている.
        # headingはよくわからない
        # 最後のvelocityは、velocity_magnitudeの1/5に等しい
        
        input_y = data["y"] # (49999, 2)
        # pos_x, pos_z
        # posは、-4.03 ~ +4.03あたりの範囲になっている.
        
        self.linear_velocities  = input_x[:,4] # (49999,)
        self.angular_velocities = input_x[:,3] # (49999,)
        
        self.pos_xs = input_y[:,0]         # (49999,) -4.5~4.5
        self.pos_zs = input_y[:,1]         # (49999,) -4.5~4.5
        self.angles = input_x[:,0] * np.pi # (49999,) -pi~pi
        
    def prepare(self, place_cells, hd_cells):
        data_size = self.linear_velocities.shape[0]

        # Prepare inputs data
        self.inputs = np.empty([data_size, 3], dtype=np.float32)
        
        self.inputs[:,0] = self.linear_velocities
        self.inputs[:,1] = np.cos(self.angular_velocities)
        self.inputs[:,2] = np.sin(self.angular_velocities)

        # Prepare outputs data
        self.place_outputs = np.empty([data_size, place_cells.cell_size])
        self.hd_outputs    = np.empty([data_size, hd_cells.cell_size])
        
        for i in range(data_size):
            pos = (self.pos_xs[i], self.pos_zs[i])
            self.place_outputs[i,:] = place_cells.get_activation(pos)
            self.hd_outputs[i,:]    = hd_cells.get_activation(self.angles[i])


    def get_train_batch(self, batch_size, sequence_length):
        episode_size = (self.linear_velocities.shape[0]+1) // EPISODE_LENGTH

        inputs_batch        = np.empty([batch_size,
                                        sequence_length,
                                        self.inputs.shape[1]])
        place_outputs_batch = np.empty([batch_size,
                                        sequence_length,
                                        self.place_outputs.shape[1]])
        hd_outputs_batch    = np.empty([batch_size,
                                        sequence_length,
                                        self.hd_outputs.shape[1]])

        place_init_batch = np.empty([batch_size,
                                     self.place_outputs.shape[1]])
        hd_init_batch    = np.empty([batch_size,
                                     self.hd_outputs.shape[1]])
        
        for i in range(batch_size):
            episode_index = np.random.randint(0, episode_size)
            pos_in_episode = np.random.randint(0, episode_size-(sequence_length+1))
            if episode_index == episode_size-1 and \
               pos_in_episode == episode_size-(sequence_length+1)-1:
                # The last espide 1 step shorter than others
                pos_in_episode -= 1
            pos = episode_index * EPISODE_LENGTH + pos_in_episode
            inputs_batch[i,:,:]        = self.inputs[pos:pos+sequence_length,:]
            place_outputs_batch[i,:,:] = self.place_outputs[pos+1:pos+sequence_length+1,:]
            hd_outputs_batch[i,:,:]    = self.hd_outputs[pos+1:pos+sequence_length+1,:]
            
            place_init_batch[i,:]   = self.place_outputs[pos,:]
            hd_init_batch[i,:]      = self.hd_outputs[pos,:]

        return inputs_batch, place_outputs_batch, hd_outputs_batch, \
            place_init_batch, hd_init_batch

    def get_confirm_index_size(self, batch_size, sequence_length):
        # total episode size (=125)
        episode_size = (self.linear_velocities.shape[0]+1) // EPISODE_LENGTH
        # sequence size per one episode (=4)
        sequence_per_episode = EPISODE_LENGTH // sequence_length
        return (episode_size * sequence_per_episode // batch_size) - 1

    def get_confirm_batch(self, batch_size, sequence_length, index):
        episode_size = (self.linear_velocities.shape[0]+1) // EPISODE_LENGTH
        
        inputs_batch     = np.empty([batch_size,
                                     sequence_length,
                                     self.inputs.shape[1]])
        place_init_batch = np.empty([batch_size,
                                     self.place_outputs.shape[1]])
        hd_init_batch    = np.empty([batch_size,
                                     self.hd_outputs.shape[1]])
        place_pos_batch  = np.empty([batch_size, sequence_length, 2])

        sequence_per_episode = EPISODE_LENGTH // sequence_length

        sequence_index = index * batch_size
        
        for i in range(batch_size):
            episode_index = sequence_index // sequence_per_episode
            pos_in_episode = (sequence_index % sequence_per_episode) * sequence_length
            pos = episode_index * EPISODE_LENGTH + pos_in_episode
            inputs_batch[i,:,:]        = self.inputs[pos:pos+sequence_length,:]
            place_init_batch[i,:]      = self.place_outputs[pos,:]
            hd_init_batch[i,:]         = self.hd_outputs[pos,:]
            # TODO: これだと最後がずれているからNG
            place_pos_batch[i,:,0]     = self.pos_xs[pos+1:pos+sequence_length+1]
            place_pos_batch[i,:,1]     = self.pos_zs[pos+1:pos+sequence_length+1]
            sequence_index += 1
        
        return inputs_batch, place_init_batch, hd_init_batch, place_pos_batch
