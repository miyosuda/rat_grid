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
        # velocityは、velocity_magnitudeの1/5に等しい
        
        input_y = data["y"] # (49999, 2)
        # pos_x, pos_z
        # posは、-4.03 ~ +4.03あたりの範囲になっている.
        
        self.velocity_xs = input_x[:,1]         # (49999,)
        self.velocity_zs = input_x[:,2]         # (49999,)
        self.anglular_velocities = input_x[:,3] # (49999,)
        
        self.pos_xs = input_y[:,0]         # (49999,) -4.5~4.5
        self.pos_zs = input_y[:,1]         # (49999,) -4.5~4.5
        self.angles = input_x[:,0] * np.pi # (49999,) -pi~pi
