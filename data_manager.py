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
