# CRM-DiMP

This project hosts the code for implementing the CRM-DiMP algorithm for visual tracking, as presented in our paper:

```
Consistent Representation Mining for Multi-Drone Single Object Tracking
```

## Tracker
#### CRM-DiMP ####

Aerial tracking has received growing attention due to its widespread practical applications.
However, single-view aerial trackers are still limited by challenges such as severe appearance variations and occlusions.
Existing multi-view trackers utilize cross-drone information to solve the above issues, but suffer from overcoming heterogenous differences in this information.
In this paper, we propose the novel Transformer-based consistent representation mining (CRM) module to capture invariant target information and suppress the heterogenous differences in cross-drone information.
First, CRM divides the heterogenous input into regions and measures semantic relevance by modeling the relations between regions.
Then reliable target regions are roughly localized by selecting the top k strongly relevant regions.
Next, the global perception is performed on these reliable regions via multi-head sparse self-attention, further enhancing the understanding of the target and suppressing background regions.
In particular, CRM, as a plug-and-play module, can be flexibly embedded into different tracking frameworks (CRM-Siam and CRM-DiMP).
Besides, the multi-view correction strategy is designed to ensure the timely correction of multi-view information and the full utilization of its own information.
Extensive experiments are conducted on the multi-drone dataset, MDOT.
The results show that the CRM-assisted trackers effectively improve the accuracy and robustness of the multi-drone tracking system and outperform other outstanding trackers.

[**[Models and Raw Results]**](https://github.com/xyl-507/CRM/releases/tag/downloads)

[**[Models and Raw Results (Baidu)]**](https://pan.baidu.com/s/15ntlgipFTmzKDclilrEg1A?pwd=1234)

### Multi-UAV Tracking

| Model(arch+backbone)  | MDOT (Suc./Pre.)   | Drone1 (Suc./Pre.)| Drone2 (Suc./Pre.) |
| --------------------  | :----------------: | :---------------: | :---------------:  |
| Super-DiMP(baseline)  |    0.506/0.728     |    0.536/0.769    |    0.476/0.687     |
| CRM-DiMP              |    0.524/0.759     |    0.535/0.774    |    0.513/0.743     |

## Installation
This document contains detailed instructions for installing the necessary dependencied for **CRM-DiMP **. The instructions 
have been tested on Ubuntu 18.04 system.

#### Install dependencies
* Create and activate a conda environment 
    ```bash
    conda create -n CRM-DiMP  python=3.7
    conda activate CRM-DiMP 
    ```  
* Install PyTorch
    ```bash
    conda install -c pytorch pytorch=1.8.0 torchvision=0.9.0 cudatoolkit=10.2
    ```  

* Install other packages
    ```bash
    conda install matplotlib pandas tqdm
    pip install opencv-python tb-nightly visdom scikit-image tikzplotlib gdown
    conda install cython scipy
    sudo apt-get install libturbojpeg
    pip install pycocotools jpeg4py
    pip install wget yacs
    pip install shapely==1.6.4.post2
    ```  
* Setup the environment                                                                                                 
Create the default environment setting files.

    ```bash
    # Change directory to <PATH_of_CRM-DiMP>
    cd CRM-DiMP
    
    # Environment settings for pytracking. Saved at pytracking/evaluation/local.py
    python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
    
    # Environment settings for ltr. Saved at ltr/admin/local.py
    python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
    ```
You can modify these files to set the paths to datasets, results paths etc.
* Add the project path to environment variables  
Open ~/.bashrc, and add the following line to the end. Note to change <path_of_CRM-DiMP> to your real path.
    ```
    export PYTHONPATH=<path_of_CRM-DiMP>:$PYTHONPATH
    ```
* Download the pre-trained networks   
Download the network for [CRM-DiMP](https://pan.baidu.com/s/15ntlgipFTmzKDclilrEg1A?pwd=1234)
and put it in the directory set by "network_path" in "pytracking/evaluation/local.py". By default, it is set to 
pytracking/networks.

## Quick Start
#### Traning
* Modify [local.py](ltr/admin/local.py) to set the paths to datasets, results paths etc.
* Runing the following commands to train the CRM-DiMP. You can customize some parameters by modifying [CRM-DiMP.py](ltr/train_settings/CRM-DiMP/CRM-DiMP.py)
    ```bash
    conda activate CRM-DiMP
    cd CRM-DiMP/ltr
    python run_training.py dimp super_dimp
    ```  

#### Test

* CUDA_VISIBLE_DEVICES=1
    ```bash
    python pytracking/run_tracker.py atom default --dataset_name uav --sequence bike1 --debug 0 --threads 0
    python pytracking/run_experiment.py myexperiments uav_test_xyl --debug 0 --threads 0
    ```

#### Evaluation
* You can use [pytracking](pytracking) to test and evaluate tracker. 
The results might be slightly different with [PySOT](https://github.com/STVIR/pysot) due to the slight difference in implementation (pytracking saves results as integers, pysot toolkit saves the results as decimals).
  

### Acknowledgement
The code based on the [PyTracking](https://github.com/visionml/pytracking),
[Biformer](https://ieeexplore.ieee.org/document/10203555).
We would like to express our sincere thanks to the contributors.
