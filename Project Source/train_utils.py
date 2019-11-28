####################################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco, Davide Maltoni, Lorenzo Pellegrini                        #
#                                                                                                  #
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file #
####################################################################################################

""" A few set of methods which can be used for training a caffe model from
    a memory input layer. See the __main__ for an usage example. """

import numpy as np
import matplotlib.pyplot as plt

import sys
from enum import Enum
from hashlib import md5
import os, time

import visualization

# Global data structures
strategy = Enum('strategy', 'SST_A_D')
backend = Enum('backend', 'CPU GPU')

from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf


def modify_net_prototxt(net_prototxt, new_net_prototxt):
    # Reads net parameters from prototxt
    net_param = caffe_pb2.NetParameter()
    with open(net_prototxt) as f:
        txtf.Merge(str(f.read()), net_param)
    for l in net_param.layer:
        # Comment / uncomment / adapt the following blocks as needed

        # For instance: this freezes all convolutions (except 1x1)
        #if l.type == "Convolution" and l.convolution_param.kernel_size[0] > 1:
        #    l.param[0].lr_mult = 0.0
        #    l.param[1].lr_mult = 0.0

        # ... this one sets the lr multiplier all convolutions/fully connected layers
        #if l.type in ["Convolution", "InnerProduct"]: # and l.name != "mid_fc8":
        #    l.param[0].lr_mult = 1.0
        #    l.param[1].lr_mult = 1.0

        # ... this one replaces the type of convolution layers (only if group != 1)
        #if l.type == "Convolution":
        #    if l.convolution_param.group !=1:
        #        l.type = "DepthwiseConvolution"

        # Sets the hyperparameters of Batch ReNorm layers
        if l.type == "BatchReNorm":
            # For batch 0
            #l.batch_renorm_param.step_to_init = 48
            #l.batch_renorm_param.step_to_r_max = 96
            #l.batch_renorm_param.step_to_d_max = 96
            #l.batch_renorm_param.r_max = 3
            #l.batch_renorm_param.d_max = 5

            # For next batches
            l.batch_renorm_param.step_to_init = 0
            l.batch_renorm_param.step_to_r_max = 1
            l.batch_renorm_param.step_to_d_max = 1
            l.batch_renorm_param.moving_average_fraction = 0.9999

            # For 79 e 196 scenarios
            l.batch_renorm_param.r_max = 1.25
            l.batch_renorm_param.d_max = 0.5

            # For 391 scenario
            #l.batch_renorm_param.r_max = 1.5
            #l.batch_renorm_param.d_max = 2.5


        #if l.type == "Scale":
        #    l.param[0].lr_mult = 0.0
        #    l.param[1].lr_mult = 0.0

        #if l.type == "Convolution" and l.convolution_param.kernel_size[0] > 1:
        #    l.param[0].lr_mult = 0.0
        #if l.type == "DepthwiseConvolution" and l.convolution_param.kernel_size[0] > 1:
        #    l.param[0].lr_mult = 0.0
        #if l.type == "Convolution" and l.convolution_param.kernel_size[0] == 1:
        #    l.param[0].lr_mult = 0.0


    with open(new_net_prototxt, 'w') as f:
       f.write(str(net_param))


def extract_minibatch_size_from_prototxt_with_input_layers(net_prototxt):
    net_param = caffe_pb2.NetParameter()
    with open(net_prototxt) as f:
        txtf.Merge(str(f.read()), net_param)
    test_minibatch_size = 0   # Maybe is not present in the prototxt
    for layer in net_param.layer:   # Search for 'data' layers
        if layer.type == 'Input' and layer.name == 'data' and len(layer.include) == 1:
            if layer.include[0].phase == 0:   # TRAIN phase
                train_minibatch_size = layer.input_param.shape[0].dim[0]
            elif layer.include[0].phase == 1:  # TEST phase
                test_minibatch_size = layer.input_param.shape[0].dim[0]
    return train_minibatch_size, test_minibatch_size


def net_use_target_vectors(net_prototxt):
    net_param = caffe_pb2.NetParameter()
    with open(net_prototxt) as f:
        txtf.Merge(str(f.read()), net_param)
    for layer in net_param.layer:   # Search for 'data' layers
        if layer.type == 'Input' and layer.name == 'target':
            return True
    return False


def shuffle_in_unison(dataset, seed):
    """ This method is used to shuffle more lists in unison. """

    np.random.seed(seed)
    rng_state = np.random.get_state()
    new_dataset = []
    for x in dataset:
        new_dataset.append(np.random.permutation(x))
        np.random.set_state(rng_state)
    return new_dataset


def preprocess_image(img):
    """ Pre-process images like caffe. It may be need adjustements depending
        on the pre-trained model since it is training dependent. """

    # We assume img is a 3-channel image loaded with plt.imread
    # So img has shape (h, w, c)

    # First, scale each pixel to 255
    img = img * 255

    # Swap RGB to BRG
    img = img[:, :, ::-1]

    # Subtract channel average
    img[:, :, 0] -= 104
    img[:, :, 1] -= 117
    img[:, :, 2] -= 123

    # Swap channel dimensions to fit the caffe format (c, h, w)
    img = np.transpose(img, (2, 0, 1))
    return img


def softmax(x):
    """ Compute softmax values for each sets of scores in x. """

    f = x - np.max(x)
    return np.exp(f) / np.sum(np.exp(f), axis=1, keepdims=True) # safe to do, gives the correct answer
    # return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def count_lines(fpath):
    """ Count line in file. """

    num_imgs = 0
    with open(fpath, 'r') as f:
        for line in f:
            if '/' in line:
                num_imgs += 1
    return num_imgs

def count_lines_in_batches(batch_count, fpath):
    lines = np.zeros(batch_count)
    for batch in range(batch_count):
        batch_filelist = fpath.replace('XX', str(batch).zfill(2))
        lines[batch] = count_lines(batch_filelist)
    return lines

def pad_data(x, y, mb_size):
    """ Padding data to suit the mini-batch size. """

    # computing test_iters
    n_missing = x.shape[0] % mb_size
    if n_missing > 0:
        surplus = 1
    else:
        surplus = 0
    it = x.shape[0] // mb_size + surplus

    # padding data to fix batch dimentions
    if n_missing > 0:
        n_to_add = mb_size - n_missing
        x = np.concatenate((x[:n_to_add], x))
        y = np.concatenate((y[:n_to_add], y))

    return x, y, it

def pad_data_single(x, mb_size):
    """ Padding data to suit the mini-batch size. """

    # Compute test_iters
    n_missing = x.shape[0] % mb_size
    if n_missing > 0:
        surplus = 1
    else:
        surplus = 0
    it = x.shape[0] // mb_size + surplus

    # Pad data to fix batch dimentions
    if n_missing > 0:
        n_to_add = mb_size - n_missing
        x = np.concatenate((x[:n_to_add], x))

    return x, it


def get_data(fpath, bpath, snap_dir, on_the_fly=False, verbose=False):
    """ Get data and label from disk. """

    # Pre-allocate numpy arrays
    num_imgs = count_lines(fpath)
    x = np.zeros((num_imgs, 3, 128, 128), dtype=np.float32)
    y = np.zeros(num_imgs, dtype=np.int8)

    # If we do not process data on the fly we check if the same train filelist
    # has been already processed and saved. If so, we load it directly.
    # In either case we end up returning x and y, as the full training set and
    # respective labels.
    hexdigest = md5(fpath.encode()).hexdigest()
    file_path_x = snap_dir + hexdigest + '_x.np'
    file_path_y = snap_dir + hexdigest + '_y.np'
    if not on_the_fly and os.path.exists(file_path_x) and os.path.exists(file_path_y):
        start = time.time()
        if verbose:
            print ("Loading data from binary dataset...")
        with open(file_path_x, 'rb') as f:
            x = np.fromfile(f, dtype = np.float32).reshape(num_imgs, 3, 128, 128)
        with open(file_path_y, 'rb') as f:
            y = np.fromfile(f, dtype = np.int8)
        if verbose:
            print ("Done in {} sec.".format(time.time()-start))
    else:
        start = time.time()
        print ("Loading image files ...")
        with open(fpath, 'r') as f:
            for i, line in enumerate(f):
                path, label = line.split()
                if verbose:
                    # print ("\r" + bpath + path + " processed: " + str(i + 1),)
                    if ((i % 1000) == 0):
                        print(i)
                image = plt.imread(bpath + path)
                x[i] = preprocess_image(image)
                y[i] = int(label)
            if verbose:
                print ("{} image files loaded and preprocessed in {} sec.".format(x.shape[0], time.time()-start))

            if not on_the_fly:
                start = time.time()
                if verbose:
                    print ("Saving data as binary dataset ...")
                with open(file_path_x, 'wb') as g:
                    x.tofile(g)   # bx = x.astype(np.uint8)  can be evaluated
                with open(file_path_y, 'wb') as g:
                    y.tofile(g)
                if verbose:
                    print ("Done in {} sec.".format(time.time()-start))
    return x, y


def test_network_with_accuracy_layer(solver, test_x, test_y, test_iterat, test_minibatch_size, final_pred_layer = 'mid_fc8', compute_test_loss = False, return_prediction = False):
    test_acc = 0
    test_loss = 0
    pred_y = np.zeros(test_y.shape[0],int)
    for test_it in range (test_iterat):
        start = test_it * test_minibatch_size
        end = (test_it + 1) * test_minibatch_size
        solver.test_nets[0].blobs['data'].data[...] = test_x[start:end]
        solver.test_nets[0].blobs['label'].data[...] = test_y[start:end]
        solver.test_nets[0].forward()
        acc = solver.test_nets[0].blobs['accuracy'].data
        test_acc += acc
        if compute_test_loss:
            loss = solver.test_nets[0].blobs['loss'].data
            test_loss += loss
        if return_prediction:
            pred_y[start:end] = solver.test_nets[0].blobs[final_pred_layer].data.argmax(1).flatten()   # need to flatten because some pred levels have size (N, 1, 1)
    test_acc /= test_iterat
    test_loss /= test_iterat
    return test_acc, test_loss, pred_y


def test_network_with_predictions_labels(net, test_x, test_y, test_iterat, test_minibatch_size, last_fc_layer = 'mid_fc8', return_prediction = False):
    correct = 0
    test_acc = 0
    pred_y = np.zeros(test_y.shape[0],int)
    for test_it in range (test_iterat):
        start = test_it * test_minibatch_size
        end = (test_it + 1) * test_minibatch_size
        net.blobs['data'].data[...] = test_x[start:end]
        net.blobs['label'].data[...] = test_y[start:end]
        net.forward()
        correct += sum(net.blobs[last_fc_layer].data.argmax(1) == net.blobs['label'].data)
        if return_prediction:
            pred_y[start:end] = net.blobs[last_fc_layer].data.argmax(1)
    test_acc = correct / (test_forward_steps*test_minibatch_size)
    return test_acc, pred_y


def compute_one_hot_vectors(train_y, class_count):
    target_y = np.zeros((train_y.shape[0], class_count), dtype=np.float32)
    target_y[np.arange(train_y.shape[0]),train_y]=1
    return target_y


def predict_labels_temporal_coherence(solver, train_x, train_y, num_classes, train_iteration_per_epoch, train_minibatch_size, strategy, confidence_thr):

    # Pre-allocate numpy arrays
    probs = np.zeros((train_iteration_per_epoch, train_minibatch_size, num_classes), dtype=np.float32)
    y_hat = np.zeros((train_iteration_per_epoch, train_minibatch_size))
    predicted_probs = np.zeros((train_iteration_per_epoch * train_minibatch_size, num_classes), dtype=np.float32)
    predicted_y = np.zeros((train_iteration_per_epoch * train_minibatch_size), dtype=np.float32)

    y_max = np.zeros((train_iteration_per_epoch, train_minibatch_size))

    for iter in range(train_iteration_per_epoch):
        start = iter * train_minibatch_size
        end = (iter + 1) * train_minibatch_size
        solver.net.blobs['data'].data[...] = train_x[start:end]
        solver.net.blobs['label'].data[...] = train_y[start:end]
        solver.net.forward()
        logits = solver.net.blobs['mid_fc8'].data
        probs[iter] = softmax(logits)                   # Probabilities for each class
        y_hat[iter] = np.argmax(probs[iter], axis=1)    # The winning class labels
        y_max[iter] = np.max(probs[iter], axis=1)    # The winning class labels

    # Compute new labels depending on the strategy
    idxs_to_remove = []

    if strategy == 'SST_A_D':
        # SST_A_D is the semi-supervised strategy based on temporal coherence

        y_hat = y_hat.flatten()
        probs = probs.reshape(train_iteration_per_epoch * train_minibatch_size, num_classes)
        for i in range(y_hat.shape[0]):
            # the first stay the same (not influencing current solution)
            if i == 0:
                predicted_y[i] = y_hat[i]
                predicted_probs[i] = probs[i]
            else:
                predicted_probs[i] = (predicted_probs[i - 1] + probs[i]) / 2
                if np.max(predicted_probs[i]) > confidence_thr:
                    predicted_y[i] = np.argmax(predicted_probs[i])
                else:
                    predicted_y[i] = 100
                    idxs_to_remove.append(i)

        # Removing examples below the threshold
        print ('Examples below confidence thr: {:.2f} %'.format(len(idxs_to_remove) / float(train_y.shape[0]) * 100))

        #predicted_y = np.delete(predicted_y, idxs_to_remove)
        #train_x = np.delete(train_x, idxs_to_remove, axis=0)

        ## Reading the same patterns above the threshold to fit the original training size.
        #n_missing = y_hat.shape[0] - predicted_y.shape[0]
        #n_adds = n_missing // predicted_y.shape[0] + 1

        #app_y = predicted_y
        #app_x = train_x
        #for i in range(n_adds):
        #    app_y = np.concatenate((app_y, predicted_y))
        #    app_x = np.concatenate((app_x, train_x))
        #predicted_y = app_y
        #train_x = app_x

        ## delete extra
        #n_more = predicted_y.shape[0] - train_y.shape[0]
        #train_x = train_x[:-n_more]
        #predicted_y = predicted_y[:-n_more]

    else:
        print ("strategy not recognized!")
        sys.exit(0)

    return train_x, predicted_y

# Copies from rand_w the output layer weights of current classes and sets a negative value for unseen classes
def dynamic_head_expansion(net, layers_to_copy, num_classes, train_y, rand_w):
    class_to_consolidate = np.unique(train_y.astype(np.int))
    min_class = class_to_consolidate.min()
    max_class = class_to_consolidate.max()
    for layer in layers_to_copy:
        for c in range(min_class, max_class+1):
            (w, b) = rand_w[layer,c]
            net.params[layer][0].data[c] = w
        for c in range(max_class+1, num_classes):
            net.params[layer][0].data[c] = -100


def stats_compute_param_change_and_update_prev(net, prev_param, batch_num, change):
    for layer_name, param in net.params.items():
        has_bias = True if len(param)>1 else False
        layer_pos = list(net._layer_names).index(layer_name)
        batch_norm = True if net.layers[layer_pos].type == 'BatchNorm' else False
        change[batch_num, layer_name, 0]=0
        change[batch_num, layer_name, 1]=0
        if (not batch_norm): change[batch_num, layer_name, 0] = np.average(np.abs(param[0].data - prev_param[layer_name, 0]))
        #print(layer_name, change[batch_num, layer_name, 0])
        if has_bias and (not batch_norm): change[batch_num, layer_name, 1] = np.average(np.abs(param[1].data - prev_param[layer_name, 1]))
        prev_param[layer_name, 0] = np.copy(param[0].data)
        if has_bias: prev_param[layer_name, 1] = np.copy(param[1].data)
    return change


def stats_initialize_param(net):
    init_param = {}
    for layer_name, param in net.params.items():
        has_bias = True if len(param)>1 else False
        init_param[layer_name, 0] = np.copy(param[0].data)
        if has_bias: init_param[layer_name, 1] = np.copy(param[1].data)
    return init_param


def stats_normalize(net, last_param, batch_count, change, normalize):
    if not normalize: return
    for layer_name, param in net.params.items():
        has_bias = True if len(param)>1 else False
        norm_w = np.average(np.abs(last_param[layer_name, 0]))
        if has_bias: norm_b = np.average(np.abs(last_param[layer_name, 1]))
        #norm_w2 = np.average(np.abs(param[0].data))  # stessa cosa di cui sopra
        #if has_bias: norm_b2 = np.average(np.abs(param[1].data))
        #print(layer_name, norm_w)
        for batch_num in range(batch_count):
            if norm_w > 0: change[batch_num, layer_name, 0] /= norm_w
            if has_bias and norm_b > 0: change[batch_num, layer_name, 1] /= norm_b

import os

def gradient_stats(prediction_level, iter, net, train_y, start, end):

    global gradient_stat_file
    if iter == 0:
        gradient_stat_file = open('GradientStats.txt','w')

    labels = train_y.astype(np.int8)

    # Gradient
    true_class_gradients = net.blobs[prediction_level].diff[list(range(end-start)),labels[start:end]]
    mean_abs_grad_true_class = np.average(np.abs(true_class_gradients))
    mean_abs_grad_all = np.average(np.abs(net.blobs[prediction_level].diff))

    # Activations
    true_class_logit_activations = net.blobs[prediction_level].data[list(range(end-start)),labels[start:end]]
    true_class_softmax_activations = net.blobs["softmax"].data[list(range(end-start)),labels[start:end]]
    mean_logit_activ = np.average(true_class_logit_activations)
    mean_softmax_activ = np.average(true_class_softmax_activations)

    gradient_stat_file.write('{:d} '.format(iter) + '{:8.6f}'.format(mean_abs_grad_all) + ' {:8.6f}'.format(mean_abs_grad_true_class) +
        ' {:8.6f}'.format(mean_logit_activ) + ' {:8.6f}\n'.format(mean_softmax_activ))

    gradient_stat_file.flush()

from shutil import copyfile

def reduce_filelist(sourcepath, destpath, keep1every):
    assert os.path.exists(sourcepath), "File does not exist!"
    with open(sourcepath, 'r') as rf:
        with open(destpath, 'w') as wf:
            for i, line in enumerate(rf):
                if (i%keep1every == 0):
                    wf.write(line)
            wf.close()
        rf.close()

def _print_bn_stats(net):
    div = net.params['conv1/bn'][2].data[0]
    conv1mean = net.params['conv1/bn'][0].data[0] / div
    conv1devstd = net.params['conv1/bn'][1].data[0] / div
    scale = net.params['conv1/scale'][0].data[0]
    bias = net.params['conv1/scale'][1].data[0]
    print('BatchNorm Conv1/bn Channel 0: Mean {:e}, Var {:e}, Count {:.3f}'.format(conv1mean, conv1devstd, div))
    print('       Conv1/Scale Channel 0: Scale {:e}, Bias {:e}'.format(scale, bias))

def print_bn_stats(net):
    div = net.params['conv1/brn'][2].data[0]
    conv1mean = net.params['conv1/brn'][0].data[0] / div
    conv1devstd = net.params['conv1/brn'][1].data[0] / div
    iter = net.params['conv1/brn'][3].data[0]
    scale = net.params['conv1/scale'][0].data[0]
    bias = net.params['conv1/scale'][1].data[0]
    print('BatchNorm Conv1/brn Channel 0: Mean {:e}, Var {:e}, Count {:.3f}, Iter {:f}'.format(conv1mean, conv1devstd, div, iter))
    print('       Conv1/Scale Channel 0: Scale {:e}, Bias {:e}'.format(scale, bias))
