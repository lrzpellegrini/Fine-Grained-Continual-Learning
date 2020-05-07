# Fine-Grained Continual Learning

This is the original Caffe implementation of **AR1\*** and **CWR\*** Continual Learning techniques.

A custom Caffe distribution packaged as a Docker image is used. More info and source code can be found [here](https://github.com/lrzpellegrini/CI-Customized-BVLC-caffe-docker). The Docker image is already available on the Docker Hub [here](https://hub.docker.com/r/lrzpellegrini/unibo_milab_caffe).

See the [Running the experiments](#running-the-experiments) section for a detailed guide on how to reproduce our experiments.

An official **PyTorch implementation** of the **AR1\*** and **CWR\*** algorithms is also available [here](https://github.com/vlomonaco/ar1-pytorch/)!

## Reference

Our article *"Rehearsal-Free Continual Learning over Small Non-I.I.D. Batches"* is now available [here](https://arxiv.org/abs/1907.03799)!

    @InProceedings{lomonaco2019nicv2,
	   title = {Rehearsal-Free Continual Learning over Small Non-I.I.D. Batches},
	   author = {Vincenzo Lomonaco and Davide Maltoni and Lorenzo Pellegrini},
	   journal = {1st Workshop on Continual Learning in Computer Vision at CVPR2020},
	   url = "https://arxiv.org/abs/1907.03799",
	   year = {2019}
	}

## Running the experiments
Some helper scripts are provided under the `Run experiments` folder.

You can run an experiment as follows.
    
1. Install the Nvidia Docker Toolkit from [here](https://github.com/NVIDIA/nvidia-docker)

2. Move inside the `Run experiments` folder:

```bash
cd "Run experiments"
```

3. Prepare the project source and create the bash script. This can be achieved by issuing the following command:

```bash
python prepare_experiment.py method scenario path-to-core50 [--nvidia_docker x]
```

where method can be "CWR", "AR1" or "Naive" and  scenario can be "79", "196" or "391". You can also execute the script with a single argument "-h" to view a description of the expected parameters.

You can set the desidered Nvidia Docker run method by passing either:
  - "--nvidia\_docker nvidia-docker2" [more info here](https://github.com/nvidia/nvidia-docker/wiki/Installation-\(version-2.0\))
  - "--nvidia\_docker nvidia-container-toolkit" [more info here](https://github.com/nvidia/nvidia-docker/wiki/Installation-\(Native-GPU-Support\))
as an argument. Defaults to nvidia-docker2.

When passing the "path-to-core50" argument, make sure that the selected folder contains the following content:
  - A file named `core50_labels.txt`, containing the Core50 labels. Can be downloaded [here](https://vlomonaco.github.io/core50/data/core50_class_names.txt)
  - A folder named `core50_128x128` containing the 128x128 version of the CORe50 dataset. Can be downloaded [here](http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip)
    
4. Running the python script will have the following effects:
    - Copy the correct prototxt files inside the `Project Source/NIC_v2/NIC_v2_X` folder
    - Create a proper `exp_configuration.json"` file inside the `Project Source` folder
    - Create a `run_experiment.sh` file inside the `Run experiments` folder. Should be already executable when created
  
5. Execute the `run_experiment.sh` script as follows:

```bash
./run_experiment.sh
```

This will run the experiment on "run0" inside our docker image in interactive mode (issuing CTRL+C or closing the terminal will terminate the experiment).

## Content

The content can be summarized as follows:

- The project source code (`Project Source` folder)
     -  `inc_training_Core50.py` contains the entry point;
     -  `nicv2_configuration.py` contains the experiment configuration loader;
     - The filelists used (`batch_filelists` subdirectory)
        - As explained in our paper, we used a reduced version of the original Core50 test set (`test_filelist_20.txt`);
        - For the proposed scenarios (NICv2 79, 196, 391) we provide a separate folder, each containing a sub-directory for each run. We used all the provided 10 runs in order to obtain the average test accuracy curves reported in our paper;
    - A MobileNetV1 pretrained with ImageNet (`models/MobileNetV1.caffemodel`)
    - The prototxt(s) describing the solvers and nets (`NIC_v2` folder)
- A set of configuration scripts to run the experiments (`Run experiments` folder)

## Core50 Dataset
The Core50 Dataset can be downloaded from <https://vlomonaco.github.io/core50/index.html#download>
In our test we used the 128x128 version, zip archive.
