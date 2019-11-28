####################################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco, Davide Maltoni, Lorenzo Pellegrini                        #
#                                                                                                  #
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file #
####################################################################################################

import shutil
import json
from pathlib import Path
import os
import stat


CONF_NIC_BASE = {
    'model': 'MobileNetV1',
    'db_path': '/datasets/core50_128x128/',   # Location of patterns and filelists
    'class_labels': '/datasets/core50_labels.txt',
    'exp_path': './Tmps/',   # Location of snapshots, temp binary database, logfiles, etc.
    'solver_file_first_batch': '---',
    'solver_file': '---',
    'init_weights_file': './models/X.caffemodel',
    'tmp_weights_file': '---',
    'train_filelists': './batch_filelists/sIII_v2_79/RUN_X/train_batch_XX_filelist.txt',
    'test_filelist': './batch_filelists/test_filelist_20.txt',
    'train_log_file': '---', # 'Cur.txt' is appended to create the file. 'Pre.txt' is appended when searching an old file for (optional) comparison.
    'num_classes': 50,  # Number of classes. There are 50 classes in Core50
    'num_batches': 79,  # including first one (all -> 79)
    #'num_batches': 196,  # including first one (all -> 196)
    #'num_batches': 391,  # including first one (all -> 391)
    'initial_batch': 0, # Valid values: (0, 1). 0 = include initial tuning from ImageNet, 1 = start from previously saved model after 0
    'num_epochs_first_batch': 4.0,  # Training epochs for first batch
    'num_epochs': 4.0,  # Training epochs for all the other batches
    'strategy': 'cwr+',  # Valid values: 'cwr+','naive','lwf','ewc','ar1'
    'lwf_weight': 0.1,  # Weight of previous training. If -1, computed according to pattern proportions in batches
    'ewc_clip_to': 0.001,  # Max value for fisher matrix elements (clip)
    'ewc_w': 1,  # Additional premultiply constant for the Fisher matrix elements
    'cwr_batch0_weight': 1.0,
    'cwr_nic_load_weight': True,
    'cwr_lr_mult': 1,  # Multiplies CWR layers LR by this value
    'backend': 'GPU',  # Valid values: 'GPU', 'CPU'
    'accuracy_max': 1.0,   # For plotting
    'test_interval_epochs': 4.0,  # Evaluation (and graphical plot) every (fraction of) batch epochs
    'dynamic_head_expansion': False,
    'confusion_matrix': False, # If True, a confusion matrix will be saved after each batch
    'save_ewc_histograms': False, # If True, EWC histogrames will be saved after each batch
    'compute_param_stats': False,
    'verbose': False
}


def configure_experiment(cl_method, scenario, db_path, project_root, scripts_path, nvidia_docker):
    selected_solver_prototxt_path = project_root / 'NIC_v2' / ('NIC_v2_' + scenario) / ('NIC_solver_MobileNetV1_' + cl_method + '.prototxt')
    print('Will use solver at', str(selected_solver_prototxt_path))

    target_solver_path = project_root / 'NIC_v2' / ('NIC_v2_' + scenario) / 'NIC_solver_MobileNetV1.prototxt'
    if target_solver_path.exists():
        target_solver_path.unlink()  # Remove old solver
    shutil.copy(str(selected_solver_prototxt_path), str(target_solver_path))
    print('Solver configured, generating experiment configuration...')

    experiment_prefix = 'NIC_v2'
    experiment_prototxts_dir = './' + experiment_prefix + '/' + experiment_prefix + '_' + scenario + '/'
    target_conf = dict(CONF_NIC_BASE)
    target_conf['exp_path'] += experiment_prefix + '_' + scenario + '/'
    target_conf['solver_file_first_batch'] = experiment_prototxts_dir + 'NIC_solver_X_first_batch.prototxt'
    target_conf['solver_file'] = experiment_prototxts_dir + 'NIC_solver_X.prototxt'
    target_conf['tmp_weights_file'] = target_conf['exp_path'] + 'X.caffemodel'
    target_conf['train_log_file'] = experiment_prototxts_dir + 'trainLog'
    target_conf['num_batches'] = int(scenario)
    target_conf['train_filelists'] = './batch_filelists/sIII_v2_' + scenario + '/RUN_X/train_batch_XX_filelist.txt'

    if cl_method == 'Naive':
        target_conf['num_epochs_first_batch'] = target_conf['num_epochs'] = target_conf['test_interval_epochs'] = 2.0
        target_conf['strategy'] = 'naive'
        target_conf['cwr_nic_load_weight'] = False
        target_conf['cwr_lr_mult'] = 1
    elif cl_method == 'CWR':
        target_conf['num_epochs_first_batch'] = target_conf['num_epochs'] = target_conf['test_interval_epochs'] = 4.0
        target_conf['strategy'] = 'cwr+'
        target_conf['cwr_nic_load_weight'] = True
        target_conf['cwr_lr_mult'] = 1
    elif cl_method == 'AR1':
        target_conf['num_epochs_first_batch'] = target_conf['num_epochs'] = target_conf['test_interval_epochs'] = 4.0
        target_conf['strategy'] = 'ar1'
        target_conf['cwr_nic_load_weight'] = True
        target_conf['cwr_lr_mult'] = 10
    
    json_target_file = project_root / 'exp_configuration.json'
    with open(str(json_target_file), 'w') as f:
        json.dump(target_conf, f)

    print('Expriment configuration created in', str(json_target_file))
    docker_script_file = Path.cwd() / 'run_experiment.sh'

    dataset_mount_option = ' -v "' + str(db_path.resolve()) +':/datasets:ro"'
    project_mount_option = ' -v "' + str(project_root.resolve()) +':/opt/project"'
    nvidia_docker_option = ' --runtime=nvidia'  # Old (nvidia-docker2)
    if nvidia_docker == 'new':
        nvidia_docker_option = ' --gpus \'all\''  # New (nvidia-container-toolkit)

    docker_script_content = '#!/bin/bash\n'
    docker_script_content += 'docker run --rm -it' + nvidia_docker_option + dataset_mount_option + project_mount_option + ' lrzpellegrini/unibo_milab_caffe python inc_training_Core50.py\n'

    if docker_script_file.exists():
        docker_script_file.unlink()

    with open(str(docker_script_file), 'w') as f:
        f.write(docker_script_content)

    st = os.stat(str(docker_script_file))
    os.chmod(str(docker_script_file), st.st_mode | stat.S_IEXEC)
    
    print('Docker script created. Configuration completed!')
    print('-> You can now run run_experiment.sh <-')
