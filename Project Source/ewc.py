####################################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco, Davide Maltoni, Lorenzo Pellegrini                        #
#                                                                                                  #
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file #
####################################################################################################

import numpy as np

import sys
import os, time

import train_utils

Ewc_free_layers = {}

def create_ewc_data(net):
    size = 0
    print('Creating Ewc data for Optimal params and their Fisher info')
    for param in net.params.items():
        layer_name = param[0]
        layer_pos = list(net._layer_names).index(layer_name)
        free_layer = True if net.layers[layer_pos].type in ['BatchNorm','BatchReNorm','Scale'] else False  # ['BatchNorm','Scale','Convolution']
        Ewc_free_layers[layer_name] = free_layer
        offset_start = size;
        num_weights = param[1][0].count
        size += num_weights     # first index is the blob name, second index = 0 denotes weight blob
        num_bias = param[1][1].count if len(param[1]) > 1 else 0
        size += num_bias        # first index is the blob name, second index = 1 denotes bias blob
        if len(param[1]) > 2:  # extra blob ?
            for i in range(2, len(param[1])):
                size+= param[1][i].count  # BatchNorm layer has three blobs, BatchRenorm 4, etc!
        if free_layer: print('Layer {:s}: free!'.format(layer_name))
        else: print('Layer {:s}: Weight {:d}, Bias {:d}, OffsetStart {:d}'.format(layer_name, num_weights, num_bias, offset_start))
    print('Total size {:d}'.format(size))
    # The first array retrurned is a 2D array: the first component containing the params at loss minimum, the second the normalized fisher info
    # The second array is the (unnormalized) Fisher matrix summed over batches
    return np.zeros((2,size), dtype = np.float32), np.zeros(size, dtype = np.float32)

def clear_gradients(net):
    for param in net.params.items():
        param[1][0].diff[...] = 0.0   # weight grads
        if len(param[1]) > 1:
            param[1][1].diff[...] = 0.0   # bias grads


def update_fisher(fisher, net, train_iteration_per_epoch, contrib):
    offset = 0
    for param in net.params.items():
        weights_diff = param[1][0].diff.flatten()
        fisher[offset:offset+weights_diff.size] += (contrib*np.square(weights_diff)/train_iteration_per_epoch)
        offset += weights_diff.size
        if len(param[1]) > 1:
            bias_diff = param[1][1].diff.flatten()
            fisher[offset:offset+bias_diff.size] += (contrib*np.square(bias_diff)/train_iteration_per_epoch)
            offset += bias_diff.size
        if len(param[1]) > 2:
            for i in range(2, len(param[1])):
                offset+= param[1][i].count


def update_fisher_in_cwr(fisher, net, train_iteration_per_epoch, contrib, cwr_layers):
    offset = 0
    for layer, param in net.params.items():
        if (layer in cwr_layers) or Ewc_free_layers[layer]:  # do not add constraints to fisher (leave at 0 -> free to move)
            weights_diff = param[0].diff.flatten()
            offset += weights_diff.size
            if len(param) > 1:
                bias_diff = param[1].diff.flatten()
                offset += bias_diff.size
            if len(param) > 2:
                for i in range(2, len(param)):
                    offset+= param[i].count
        else:
            weights_diff = param[0].diff.flatten()
            fisher[offset:offset+weights_diff.size] += (contrib*np.square(weights_diff)/train_iteration_per_epoch)
            offset += weights_diff.size
            if len(param) > 1:
                bias_diff = param[1].diff.flatten()
                fisher[offset:offset+bias_diff.size] += (contrib*np.square(bias_diff)/train_iteration_per_epoch)
                offset += bias_diff.size
            if len(param) > 2:
                for i in range(2, len(param)):
                    offset+= param[i].count

def update_ewc_data(ewcData, fisher, net, train_x, train_y, target_y, train_iteration_per_epoch, train_minibatch_size, batch_num, clip_to, ewc_w = 1.0, cwr_layers = None):
    contrib = 1

    # First, update the fisher info (unnormalized summed matrix)
    for iter in range(train_iteration_per_epoch):
        start = iter * train_minibatch_size
        end = (iter + 1) * train_minibatch_size
        net.blobs['data'].data[...] = train_x[start:end]
        net.blobs['label'].data[...] = train_y[start:end]
        if target_y is not None:
           net.blobs['target'].data[...] = target_y[start:end]
        clear_gradients(net)
        net.forward()
        net.backward()
        if cwr_layers == None:
            update_fisher(fisher, net, train_iteration_per_epoch, contrib)
        else:
            update_fisher_in_cwr(fisher, net, train_iteration_per_epoch, contrib, cwr_layers)

    # normalize into ewcData[1]
    ewcData[1] = (ewc_w/(batch_num+1)) * fisher
    normalized_fisher = ewcData[1]

    # MAX instead of SUM -> put fisher[...] = 0 before loop
    # ewcData[1] = np.maximum.reduce([ewcData[1], fisher])
    # normalized_fisher = ewcData[1]

    # Hard: Relu like clip
    normalized_fisher[normalized_fisher>clip_to] = clip_to   # clip to max

    # Soft: half sigmoid based [0 in 0 : 0.5 in mean : 1 in clip_to] * clip_to
    # coeff=3500
    # normalized_fisher[...] = 2*clip_to * (1/(1+np.exp(-coeff*normalized_fisher[...])) - 0.5)

    # Then extract and copy the optimal weights
    theta = ewcData[0]
    offset = 0
    theta[...]=0
    for param in net.params.items():
        weights = param[1][0].data.flatten()
        # weights[...] = 0
        theta[offset:offset+weights.size] = weights
        offset += weights.size
        if len(param[1]) > 1:
            bias = param[1][1].data.flatten()
            # bias[...] = 1
            theta[offset:offset+bias.size] = bias
            offset += bias.size
        if len(param[1]) > 2:  # extra blob ?
            for i in range(2, len(param[1])):
                offset+= param[1][i].count
