####################################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco, Davide Maltoni, Lorenzo Pellegrini                        #
#                                                                                                  #
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file #
####################################################################################################

import sys
import argparse
from pathlib import Path
import json
import experiments_configuration

PROJECT_SOURCE_PATH = Path('../Project Source/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Executes the desidered experiments', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('cl_method', help='The CL method. Valid values: Naive, CWR, AR1')
    parser.add_argument('scenario', help='The scenario. Valid values: 79, 196, 391')
    parser.add_argument('db_path', help='The path to the Core50 dataset.Check experiments_configuration.py script for more details on the expected directory structure!')
    parser.add_argument('--nvidia_docker', help='Nvidia Docker version. Valid values: "nvidia-docker2" (or "old"), "nvidia-container-toolkit" (or "new"). Defaults to nvidia-docker2\nvidia-docker2: <https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)>\nvidia-container-toolkit: <https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)#usage>')

    args = parser.parse_args()

    cl_method = args.cl_method
    if cl_method not in ['Naive', 'CWR', 'AR1']:
        print('Invalid CL method', cl_method)
        exit(1)
    
    scenario = args.scenario
    if scenario not in ['79', '196', '391']:
        print('Invalid scenario', cl_method)
        exit(1)

    db_path = Path(args.db_path)
    if not db_path.exists():
        print('Invalid dataset path', str(db_path))
        exit(1)

    nvidia_docker = 'old'
    if args.nvidia_docker is not None:
        nvidia_docker = args.nvidia_docker

    if nvidia_docker not in ['old', 'new', 'nvidia-docker2', 'nvidia-container-toolkit']:
        print('Invalid nvidia docker argument. Must be "old" or "new".')
        exit(1)
    
    if nvidia_docker == 'nvidia-docker2':
        nvidia_docker = 'old'
    elif nvidia_docker == 'nvidia-container-toolkit':
        nvidia_docker = 'new'

    scripts_path = Path.cwd()
    experiments_configuration.configure_experiment(cl_method, scenario, db_path, PROJECT_SOURCE_PATH, scripts_path, nvidia_docker)

    

    