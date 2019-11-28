####################################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco, Davide Maltoni, Lorenzo Pellegrini                        #
#                                                                                                  #
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file #
####################################################################################################

import sys, os, time
import numpy as np

# Uncomment and customize the following lines if PyCaffe needs to be loaded dinamically
# caffe_root = 'D:/CaffeInstall/'
# sys.path.insert(0, caffe_root + 'python')
os.environ["GLOG_minloglevel"] = "1"  # limit logging (0 - debug, 1 - info (still a LOT of outputs), 2 - warnings, 3 - errors)
import caffe

# For prototxt parsing
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

from sklearn.metrics import confusion_matrix

import visualization
import filelog
import train_utils
import lwf, ewc, cwr, syn
from pathlib import Path

def main_Core50(conf, run, close_at_the_end = False):

    # Prepare configurations files
    conf['solver_file_first_batch'] = conf['solver_file_first_batch'].replace('X', conf['model'])
    conf['solver_file'] = conf['solver_file'].replace('X', conf['model'])
    conf['init_weights_file'] = conf['init_weights_file'].replace('X', conf['model'])
    conf['tmp_weights_file'] = conf['tmp_weights_file'].replace('X', conf['model'])
    train_filelists = conf['train_filelists'].replace('RUN_X', run)
    test_filelist = conf['test_filelist'].replace('RUN_X', run)

    # For run 0 store/load binary files
    # For the rest of runs read single files (slower, but saves disk space)
    #run_on_the_fly = False if run == 'run0' else True
    run_on_the_fly = True
    # This is the procedure we applied to obtain the reduced test set
    # train_utils.reduce_filelist(test_filelist, test_filelist+"3", 20)

    (Path(conf['exp_path']) / 'CM').mkdir(exist_ok=True, parents=True)
    (Path(conf['exp_path']) / 'EwC').mkdir(exist_ok=True, parents=True)
    (Path(conf['exp_path']) / 'Syn').mkdir(exist_ok=True, parents=True)

    # Parse the solver prototxt
    #  for more details see - https://stackoverflow.com/questions/31823898/changing-the-solver-parameters-in-caffe-through-pycaffe
    print('Solver proto: ', conf['solver_file_first_batch'])
    solver_param = caffe_pb2.SolverParameter()
    with open(conf['solver_file_first_batch']) as f:
        txtf.Merge(str(f.read()), solver_param)
    net_prototxt = solver_param.net  # Obtains the path to the net prototxt
    print('Net proto: ',net_prototxt)

    # Obtain class labels
    if conf['class_labels'] != '':
        # More complex than a simple loadtxt because of the unicode representation in python 3
        label_str = np.loadtxt(conf['class_labels'], dtype=bytes, delimiter="\n").astype(str)

    # Obtain minibatch size from net proto
    train_minibatch_size, test_minibatch_size = train_utils.extract_minibatch_size_from_prototxt_with_input_layers(net_prototxt)
    print(' test minibatch size: ', test_minibatch_size)
    print(' train minibatch size: ', train_minibatch_size)

    # Is the network using target vectors (besides the labels)?
    need_target = train_utils.net_use_target_vectors(net_prototxt)

    # Load test set
    print ("Recovering Test Set: ", test_filelist, " ...")
    start = time.time()
    test_x, test_y = train_utils.get_data(test_filelist, conf['db_path'], conf['exp_path'], on_the_fly=run_on_the_fly, verbose = conf['verbose'])
    assert(test_x.shape[0] == test_y.shape[0])
    if conf['num_classes'] == 10:  # Checks if we are doing category-based classification
        test_y = test_y // 5
    test_y = test_y.astype(np.float32)
    test_patterns = test_x.shape[0]
    test_x, test_y, test_iterat = train_utils.pad_data(test_x, test_y, test_minibatch_size)
    print (' -> %d patterns of %d classes (%.2f sec.)' % (test_patterns, len(np.unique(test_y)), time.time() - start))
    print (' -> %.2f -> %d iterations for full evaluation' % (test_patterns / test_minibatch_size, test_iterat))

    # Load training patterns in batches (by now assume the same number in all batches)
    batch_count = conf['num_batches']
    train_patterns = train_utils.count_lines_in_batches(batch_count,train_filelists)
    train_iterations_per_epoch = np.zeros(batch_count, int)
    train_iterations = np.zeros(batch_count, int)
    test_interval_epochs = conf['test_interval_epochs']
    test_interval = np.zeros(batch_count, float)
    for batch in range(batch_count):
        train_iterations_per_epoch[batch] = int(np.ceil(train_patterns[batch] / train_minibatch_size))
        test_interval[batch] = test_interval_epochs * train_iterations_per_epoch[batch]
        if (batch == 0):
            train_iterations[batch] = train_iterations_per_epoch[batch] * conf['num_epochs_first_batch']
        else:
            train_iterations[batch] = train_iterations_per_epoch[batch] * conf['num_epochs']
        print ("Batch %2d: %d patterns, %d iterations (%d iter. per epochs - test every %.1f iter.)" \
             % (batch, train_patterns[batch],  train_iterations[batch], train_iterations_per_epoch[batch], test_interval[batch]))

    # Create evaluation points
    # -> iterations which are boundaries of batches
    batch_iter = [0]
    iter = 0
    for batch in range(batch_count):
        iter += train_iterations[batch]
        batch_iter.append(iter)

    # Calculates the iterations where the network will be evaluated
    eval_iters = [1]   # Start with 1 (insted of 0) because the test net is aligned to the train one after solver.step(1)
    for batch in range(batch_count):
        start = batch_iter[batch]
        end = batch_iter[batch+1]
        start += test_interval[batch]
        while start < end:
            eval_iters.append(int(start))
            start += test_interval[batch]
        eval_iters.append(end)

    # Iterations which are epochs in the evaluation range
    epochs_iter = []
    for batch in range(batch_count):
        start = batch_iter[batch]
        end = batch_iter[batch+1]
        start += train_iterations_per_epoch[batch]
        while start <= end:
            epochs_iter.append(int(start))
            start += train_iterations_per_epoch[batch]

    prev_train_loss = np.zeros(len(eval_iters))
    prev_test_acc = np.zeros(len(eval_iters))
    prev_exist = filelog.TryLoadPrevTrainingLog(conf['train_log_file'], prev_train_loss, prev_test_acc)
    train_loss = np.copy(prev_train_loss)  # Copying allows to correctly visualize the graph in case we start from initial_batch > 0
    test_acc = np.copy(prev_test_acc)
    train_acc = np.zeros(len(eval_iters))

    epochs_tick = False if batch_count > 30 else True  # For better visualization
    visualization.Plot_Incremental_Training_Init('Incremental Training', eval_iters, epochs_iter, batch_iter, train_loss, test_acc, 5, conf['accuracy_max'], prev_exist, prev_train_loss, prev_test_acc, show_epochs_tick = epochs_tick)
    filelog.Train_Log_Init(conf['train_log_file'])
    filelog.Train_LogDetails_Init(conf['train_log_file'])

    start_train = time.time()
    eval_idx = 0   # Evaluation iterations counter
    global_eval_iter = 0  # Global iterations counter
    first_round = True
    initial_batch = conf['initial_batch']
    if initial_batch > 0:  # Move forward by skipping unnecessary evaluation
        global_eval_iter = batch_iter[initial_batch]
        while eval_iters[eval_idx] < global_eval_iter:
            eval_idx += 1
        eval_idx += 1

    for batch in range(initial_batch, batch_count):
        print ('\nBATCH = {:2d} ----------------------------------------------------'.format(batch))

        if batch == 0:
            solver = caffe.get_solver(conf['solver_file_first_batch'])   # Load the solver for the first batch and create net(s)
            if conf['init_weights_file'] !='':
                solver.net.copy_from(conf['init_weights_file'])
                print('Network created and Weights loaded from: ', conf['init_weights_file'])
                # Test
                solver.test_nets[0].copy_from(conf['init_weights_file'])
                accuracy, _ , pred_y = train_utils.test_network_with_accuracy_layer(solver, test_x, test_y, test_iterat, test_minibatch_size, prediction_level_Model[conf['model']], return_prediction = True)

                # BatchNorm Stats
                train_utils.print_bn_stats(solver.net)

            if conf['strategy'] in ['cwr+','ar1']:
                cwr.zeros_cwr_layer_bias_lr(solver.net, cwr_layers_Model[conf['model']])
                class_updates = np.zeros(conf['num_classes'], dtype=np.float32)
                cons_w = cwr.init_consolidated_weights(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'])    # Allocate space for consolidated weights and initialze them to 0
                cwr.reset_weights(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'])   # Reset cwr weights to 0 (done here for the first batch to keep initial stats correct)

            if conf['strategy'] == 'cwr' or conf['dynamic_head_expansion'] == True:
                class_updates = np.zeros(conf['num_classes'], dtype=np.float32)
                rand_w, cons_w = cwr.copy_initial_weights(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'])    # Random values for cwr layers (since they do not exist in pretrained models)

            if conf['strategy'] in ['syn','ar1']:
                # ewcData stores optimal weights + normalized fisher; trajectory stores unnormalized summed grad*deltaW
                ewcData, synData = syn.create_syn_data(solver.net)

        elif batch == 1:
            solver = caffe.get_solver(conf['solver_file'])   # Load the solver for the next batches and create net(s)
            solver.net.copy_from(conf['tmp_weights_file'])
            print('Network created and Weights loaded from: ', conf['tmp_weights_file'])

            if conf['strategy'] in ['cwr','cwr+']:
                cwr.zeros_non_cwr_layers_lr(solver.net, cwr_layers_Model[conf['model']])   # In CWR we freeze every layer except the CWR one(s)

            # By providing a cwr_lr_mult multiplier we can use a different Learning Rate for CWR and non-CWR cwr_layers_Model
            # Note that a similar result can be achieved by manually editing the net prototxt
            if conf['strategy'] in ['cwr+', 'ar1']:
                if 'cwr_lr_mult' in conf.keys() and conf['cwr_lr_mult'] != 1:
                    cwr.zeros_cwr_layer_bias_lr(solver.net, cwr_layers_Model[conf['model']], force_weights_lr_mult = conf['cwr_lr_mult'])
                else:
                    cwr.zeros_cwr_layer_bias_lr(solver.net, cwr_layers_Model[conf['model']])

            cwr.set_brn_past_weight(solver.net, 10000)

        # Initializes some data structures used for reporting stats. Executed once (in the first round)
        if first_round:
            if batch == 1 and (conf['strategy'] in ['ewc','cwr', 'cwr+', 'syn', 'ar1']):
                print('Cannot start from batch 1 in ', conf['strategy'], ' strategy!')
                sys.exit(0)
            visualization.PrintNetworkArchitecture(solver.net)
            # if accuracy layer is defined in the prototxt also in TRAIN mode -> log also train accuracy (not in the plot)
            try:
                report_train_accuracy = True
                err = solver.net.blobs['accuracy'].num  # assume this is stable for prototxt of successive batches
            except:
                report_train_accuracy = False
            first_round = False
            if conf['compute_param_stats']:
                param_change = {}
                param_stats = train_utils.stats_initialize_param(solver.net)

        # Load training data for the current batch
        # Note that the file lists are provided in the batch_filelists folder
        current_train_filelist = train_filelists.replace('XX', str(batch).zfill(2))
        print ("Recovering training data: ", current_train_filelist, " ...")
        load_start = time.time()
        train_x, train_y = train_utils.get_data(current_train_filelist, conf['db_path'], conf['exp_path'], on_the_fly=run_on_the_fly, verbose = conf['verbose'])
        print ("Done.")
        if conf['num_classes'] == 10:  # Category based classification
            train_y = train_y // 5

        # If target values (e.g. one hot vectors) are needed we need to create them from numerical class labels
        if need_target:
            target_y = train_utils.compute_one_hot_vectors(train_y, conf['num_classes'])
            train_x, tmp_iters = train_utils.pad_data_single(train_x, train_minibatch_size)
            train_y, _ = train_utils.pad_data_single(train_y, train_minibatch_size)
            target_y, _ = train_utils.pad_data_single(target_y, train_minibatch_size)

            if batch>0 and conf['strategy'] == 'lwf':
                if conf['lwf_weight'] > 0: weight_old = conf['lwf_weight']
                else:
                    weight_old = 1 - (train_patterns[batch] / np.sum(train_patterns[0:batch+1]))
                    x_min = 2.0/3.0
                    x_max = 0.9
                    y_min = 0.45
                    y_max = 0.60
                    #
                    if weight_old > x_max: weight_old = x_max # Clip weight_old
                    weight_old = y_min + (weight_old - x_min)*(y_max-y_min)/(x_max-x_min)
                print('Lwf Past Weight: %.2f' % (weight_old))
                target_y = lwf.update_target_vectors(solver, train_x, train_y, conf['num_classes'], train_iterations_per_epoch[batch], train_minibatch_size, weight_old)

            if conf['dynamic_head_expansion'] == True:
                train_utils.dynamic_head_expansion(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'], train_y, rand_w)

            if conf['strategy'] == 'cwr' and batch > initial_batch:
                cwr.load_weights(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'], rand_w)  # Reset net weights
            if conf['strategy'] in ['cwr+','ar1'] and batch > initial_batch:
                cwr.reset_weights(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'])  # Reset weights of CWR layers to 0  (maximal head approach!)
                # Loads previously consolidated weights
                # This procedure, explained in the paper, is necessary in the NIC scenario
                if 'cwr_nic_load_weight' in conf.keys() and conf['cwr_nic_load_weight']:
                    cwr.load_weights_nic(solver.net, cwr_layers_Model[conf['model']], train_y, cons_w)

            train_x, train_y, target_y = train_utils.shuffle_in_unison((train_x, train_y, target_y), 0)
            if conf['strategy'] in ['ewc','syn','ar1'] and batch > initial_batch:
                #syn.weight_stats(solver.net, batch, ewcData, conf['ewc_clip_to'])
                # Makes ewc info available to the network for successive training
                # The 'ewc' blob will be used by our C++ code (see the provided custom "sgd_solver.cpp")
                solver.net.blobs['ewc'].data[...] = ewcData
        else:
            #TODO: review branch (is it necessary?)
            train_x, tmp_iters = train_utils.pad_data_single(train_x, train_minibatch_size)
            train_y, _ = train_utils.pad_data_single(train_y, train_minibatch_size)
            train_x, train_y = train_utils.shuffle_in_unison((train_x, train_y), 0)
            # apply temporal coherence strategy to modify labels
            if batch > 0 and conf['strategy'] != 'naive':
                train_x, train_y = train_utils.predict_labels_temporal_coherence(solver, train_x, train_y, conf['num_classes'], train_iterations_per_epoch[batch], train_minibatch_size, conf['strategy'], 0.80)
                # ATTENTION, if patterns have been removed do padding again!

        print (' -> %d patterns (of %d classes) after padding and shuffling (%.2f sec.)' % (train_x.shape[0], len(np.unique(train_y)), time.time()-load_start))
        assert(train_iterations[batch] >= tmp_iters)

        # convert labels to float32
        train_y = train_y.astype(np.float32)
        assert(train_x.shape[0] == train_y.shape[0])

        # training
        avg_train_loss = 0
        avg_train_accuracy = 0
        avg_count = 0;

        if conf['strategy'] in ['syn','ar1']:
            syn.init_batch(solver.net, ewcData, synData)

        # The main solver loop (per batch)
        it = 0
        while it < train_iterations[batch]:
            # The following part is pretty much straight-forward
            # The current batch is split in minibatches (which size was previously detected by looking at the net prototxt)
            # The minibatch is loaded in blobs 'data', 'label' and 'target'
            # a step(1) is executed (which executes forward + backward + weights update)
            it_mod = it % train_iterations_per_epoch[batch]
            start = it_mod * train_minibatch_size
            end = (it_mod + 1) * train_minibatch_size

            if conf['verbose']:
                avgl = avga = 0
                if avg_count > 0:
                    avgl = avg_train_loss / avg_count
                print ('Iter {:>4}'.format(it+1), '({:>4})'.format(global_eval_iter), ': Train Loss = {:.5f}'.format(avgl), end='', flush = True)
                if report_train_accuracy:
                    if avg_count > 0:
                        avga = avg_train_accuracy / avg_count
                    print ('  Train Accuracy = {:.5f}%'.format(avga*100), flush = True)
            else:
                print ('+', end = '', flush=True)

            # Provide data to input layers
            solver.net.blobs['data'].data[...] = train_x[start:end]
            solver.net.blobs['label'].data[...] = train_y[start:end]
            if need_target:
                solver.net.blobs['target'].data[...] = target_y[start:end]

            if conf['strategy'] in ['syn','ar1']:
                syn.pre_update(solver.net, ewcData, synData)

            # SGD by Caffe
            if conf['strategy'] in ['cwr+','cwr'] and batch > initial_batch:
                solver.net.clear_param_diffs()
                solver.net.forward()  # start=None, end=None
                solver.net.backward(end='mid_fc7')
                solver.apply_update()
            else:
                solver.step(1)
            #train_utils.print_bn_stats(solver.net)

            # If enabled saves the gradient magniture of the prediction_level stats on file
            # train_utils.gradient_stats(prediction_level_Model[conf['model']], global_eval_iter, solver.net, train_y, start, end)

            if conf['strategy'] == 'syn':
                syn.post_update(solver.net, ewcData, synData)
            if conf['strategy'] == 'ar1':
                syn.post_update(solver.net, ewcData, synData, cwr_layers_Model[conf['model']])

            global_eval_iter +=1
            avg_count +=1

            avg_train_loss += solver.net.blobs['loss'].data
            if report_train_accuracy:
                avg_train_accuracy += solver.net.blobs['accuracy'].data

            # Early stopping (a.k.a. Limited)
            if conf['strategy'] == '_syn' and avg_count > 0 and avg_train_loss/avg_count < syn.target_train_loss_accuracy_per_batch(batch):    # enable by removing "_" on demand
                it = train_iterations[batch]-1              # skip to last iter
                global_eval_iter = eval_iters[eval_idx]     # enable evaluation point now

            if global_eval_iter == eval_iters[eval_idx]:
                # Evaluation point
                if avg_count > 0:
                    avg_train_loss/= avg_count
                    avg_train_accuracy /= avg_count
                train_loss[eval_idx] = avg_train_loss
                print ('\nIter {:>4}'.format(it+1), '({:>4})'.format(global_eval_iter), ': Train Loss = {:.5f}'.format(avg_train_loss), end='', flush = True)
                if report_train_accuracy:
                    train_acc[eval_idx] = avg_train_accuracy
                    print ('  Train Accuracy = {:.5f}%'.format(avg_train_accuracy*100), end='', flush = True)

                compute_confusion_matrix = True if (conf['confusion_matrix'] and it == train_iterations[batch]-1) else False   # last batch iter

                # The following lines are executed only if this is the last iteration for the current batch
                if conf['strategy'] in ['cwr', 'cwr+', 'ar1'] and it == train_iterations[batch]-1:
                    if conf['strategy'] == 'cwr':
                        batch_weight = conf['cwr_batch0_weight'] if batch == initial_batch else 1
                        cwr._consolidate_weights_cwr(solver.net, cwr_layers_Model[conf['model']], train_y, cons_w, batch_weight, class_updates = class_updates)
                        class_updates[train_y.astype(np.int)] += 1;  # Increase weights of trained classes
                    else:
                        unique_y, y_freq = np.unique(train_y.astype(np.int), return_counts=True)
                        cwr.consolidate_weights_cwr_plus(solver.net, cwr_layers_Model[conf['model']], unique_y, y_freq, class_updates, cons_w)
                        class_updates[unique_y] += y_freq;

                    # print(class_updates)
                    cwr.load_weights(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'], cons_w)   # Load consolidated weights for testing

                accuracy, _ , pred_y = train_utils.test_network_with_accuracy_layer(solver, test_x, test_y, test_iterat, test_minibatch_size, prediction_level_Model[conf['model']], return_prediction = compute_confusion_matrix)
                test_acc[eval_idx] = accuracy*100
                print ('  Test Accuracy = {:.5f}%'.format(accuracy*100))

                # Batch(Re)Norm Stats
                train_utils.print_bn_stats(solver.net)

                visualization.Plot_Incremental_Training_Update(eval_idx, eval_iters, train_loss, test_acc)

                filelog.Train_Log_Update(conf['train_log_file'], eval_iters[eval_idx], accuracy, avg_train_loss, report_train_accuracy, avg_train_accuracy)

                avg_train_loss = 0
                avg_train_accuracy = 0
                avg_count = 0
                eval_idx+=1   # next eval

            it+=1  # next iter

        # Current batch training concluded
        if conf['strategy'] == 'ewc':
            if batch == initial_batch:
                ewcData, fisher = ewc.create_ewc_data(solver.net)   # ewcData stores optimal weights + normalized fisher; fisher store unnormalized summed fisher
            print ("Computing Fisher Information and Storing Optimal Weights...")
            ewc.update_ewc_data(ewcData, fisher, solver.net, train_x, train_y, target_y, train_iterations_per_epoch[batch], train_minibatch_size, batch, conf['ewc_clip_to'], conf['ewc_w'])
            print ("Done.")
            if conf['save_ewc_histograms']:
                visualization.EwcHistograms(ewcData, 100, save_as = conf['exp_path'] + 'EwC/F_' + str(batch) + '.png')

        if conf['strategy'] in ['syn','ar1']:
            syn.update_ewc_data(solver.net, ewcData, synData, batch, conf['ewc_clip_to'])
            if conf['save_ewc_histograms']:
                visualization.EwcHistograms(ewcData, 100, save_as = conf['exp_path'] + 'Syn/F_' + str(batch) + '.png')

        if compute_confusion_matrix:
            # Computes the confusion matrix and logs + plots it
            cnf_matrix = confusion_matrix(test_y, pred_y)
            if batch ==0:
                prev_class_accuracies = np.zeros(conf['num_classes'])
            else:
                prev_class_accuracies = current_class_accuracies
            current_class_accuracies = np.diagonal(cnf_matrix) / cnf_matrix.sum(axis = 1)
            deltas = current_class_accuracies - prev_class_accuracies
            classes_in_batch = set(train_y.astype(np.int))
            classes_non_in_batch = set(range(conf['num_classes']))-classes_in_batch
            mean_class_in_batch = np.mean(deltas[list(classes_in_batch)])
            std_class_in_batch = np.std(deltas[list(classes_in_batch)])
            mean_class_non_in_batch = np.mean(deltas[list(classes_non_in_batch)])
            std_class_non_in_batch = np.std(deltas[list(classes_non_in_batch)])
            print('InBatch -> mean =  %.2f%% std =  %.2f%%, OutBatch -> mean =  %.2f%% std =  %.2f%%' % (mean_class_in_batch*100, std_class_in_batch*100, mean_class_non_in_batch*100, std_class_non_in_batch*100))
            filelog.Train_LogDetails_Update(conf['train_log_file'], batch, mean_class_in_batch, std_class_in_batch, mean_class_non_in_batch, std_class_non_in_batch)
            visualization.plot_confusion_matrix(cnf_matrix, normalize = True, title='CM after batch: ' + str(batch), save_as = conf['exp_path'] + 'CM/CM_' + str(batch) + '.png')

        if conf['compute_param_stats']:
            train_utils.stats_compute_param_change_and_update_prev(solver.net, param_stats, batch, param_change)

        if batch == 0:
            solver.net.save(conf['tmp_weights_file'])
            print('Weights saved to: ', conf['tmp_weights_file'])
            del solver

    print('Training Time: %.2f sec' % (time.time() - start_train))

    if conf['compute_param_stats']:
        stats_normalization = True
        train_utils.stats_normalize(solver.net, param_stats, batch_count, param_change, stats_normalization)
        visualization.Plot3d_param_stats(solver.net, param_change, batch_count, stats_normalization)

    filelog.Train_Log_End(conf['train_log_file'])
    filelog.Train_LogDetails_End(conf['train_log_file'])

    visualization.Plot_Incremental_Training_End(close = close_at_the_end)


def main_Core50_multiRun(conf, runs):

    conf['confusion_matrix'] = True
    conf['save_ewc_histograms'] = False
    conf['compute_param_stats'] = False
    allfile = open(conf['train_log_file']+'All.txt', 'w')
    for r in range(runs):
        run = 'run'+str(r)
        main_Core50(conf, run, close_at_the_end = True)
        runres = filelog.LoadAccuracyFromCurTrainingLog(conf['train_log_file'])
        for item in runres:
            allfile.write("%s " % item)
        allfile.write("\n")
        allfile.flush()
    allfile.close()

if __name__ == "__main__":

    import nicv2_configuration

    ## --------------------------------------------------------------------

    prediction_level_Model = {
        'CaffeNet': 'mid_fc8',
        'Nin': 'pool4',
        'GoogleNet': 'loss3_50/classifier',
        'MobileNetV1': 'mid_fc7'
    }

    cwr_layers_Model = {
        'CaffeNet': ['mid_fc8'],
        'GoogleNet': ['loss1_50/classifier', 'loss2_50/classifier', 'loss3_50/classifier'],
        'MobileNetV1': ['mid_fc7'],
    }

    conf = nicv2_configuration.conf_NIC

    if conf['backend'] == 'GPU':
       caffe.set_device(0)
       caffe.set_mode_gpu()
    else:
       caffe.set_mode_cpu()

    if conf['strategy'] not in ['naive','cwr','cwr+','ewc','lwf','syn','ar1']:
        print("Undefined strategy!")
        sys.exit(1)

    # Single Run
    sys.exit(int(main_Core50(conf, 'run0') or 0))

    # Multi Run
    # runs = 10
    # sys.exit(int(main_Core50_multiRun(conf, runs) or 0))
