####################################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco, Davide Maltoni, Lorenzo Pellegrini                        #
#                                                                                                  #
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file #
####################################################################################################

import numpy as np

import sys
import os, time

import train_utils

def update_target_vectors(solver, train_x, train_y, num_classes, train_iteration_per_epoch, train_minibatch_size, prevmodel_weight):
    # Pre-allocate numpy arrays
    probs = np.zeros((train_iteration_per_epoch, train_minibatch_size, num_classes), dtype=np.float32)
    new_target_y = train_utils.compute_one_hot_vectors(train_y, num_classes)   # Start from one_hot

    for iter in range(train_iteration_per_epoch):
        start = iter * train_minibatch_size
        end = (iter + 1) * train_minibatch_size
        solver.net.blobs['data'].data[...] = train_x[start:end]
        # Target and labels are not necessary (we only need softmax probs)
        solver.net.forward(end = 'softmax')
        probs[iter] = solver.net.blobs['softmax'].data.reshape(train_minibatch_size, num_classes)   # Size cloud be (M, N, 1, 1) but expect (M, N)

    probs = probs.reshape(train_iteration_per_epoch * train_minibatch_size, num_classes)   # Flatten
    new_target_y = (1-prevmodel_weight) * new_target_y + prevmodel_weight * probs  # Enforce stability w.r.t. previsous batch

    return new_target_y
