# CRM-Siam

This project hosts the code for implementing the CRM-Siam algorithm of the 2024 IEEE Transactions on Circuits and Systems for Video Technology paper:
```
Consistent Representation Mining for Multi-Drone Single Object Tracking
(accepted by IEEE Transactions on Circuits and Systems for Video Technology, DOI: 10.1109/TCSVT.2024.3411301)
```

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

The paper can be downloaded from [IEEE Xplore]()

[**[Models and Raw Results]**](https://github.com/xyl-507/CRM/releases/tag/downloads)

[**[Models and Raw Results (Baidu)]**](https://pan.baidu.com/s/15ntlgipFTmzKDclilrEg1A?pwd=1234)

### Proposed modules
- `CRM` in [model](https://github.com/xyl-507/CRM/blob/main/CRM-Siam/siamban/models/MobileViTAttention.py)

### Multi-UAV Tracking

| Model                 | MDOT (Suc./Pre.)   | Drone1 (Suc./Pre.)| Drone2 (Suc./Pre.) |
| --------------------  | :----------------: | :---------------: | :---------------:  |
| SiamBAN (baseline)    |    0.394/0.570     |    0.426/0.619    |    0.363/0.521     |
| CRM-Siam              |    0.401/0.602     |    0.417/0.627    |    0.385/0.577     |


## Installation

Please find installation instructions in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using CRM-Siam

### Add CRM-Siam to your PYTHONPATH

```bash
export PYTHONPATH=/path/to/CRM-Siam:$PYTHONPATH (for linux)
set PYTHONPATH=%PYTHONPATH%;/path/to/CRM-Siam (for windows)
```


### demo

```bash
python tools/demo.py \
    --config experiments/siamban_r50_l234/config-CRM-Siam.yaml \
    --snapshot experiments/siamban_r50_l234/CRM-Siam.pth
    --video demo/bag.avi
```

### Download testing datasets

Download datasets and put them into `testing_dataset` directory. Jsons of commonly used datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI) or [BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F). If you want to test tracker on new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to setting `testing_dataset`. 

### Test tracker

```bash
cd experiments/siamban_r50_l234
python -u ../../tools/test-multi.py --snapshot CRM-Siam.pth --config config-CRM-Siam.yaml --dataset MDOT
```

The testing results will in the current directory(results/dataset/model_name/)

### Eval tracker

assume still in experiments/siamban_r50_l234

``` bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset MDOT        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'CRM*'   # tracker_name
```

###  Training :wrench:

See [TRAIN.md](TRAIN.md) for detailed instruction.


### Acknowledgement
The code based on the [PySOT](https://github.com/STVIR/pysot) , [SiamBAN](https://github.com/hqucv/siamban) ,
[Biformer](https://ieeexplore.ieee.org/document/10203555).
We would like to express our sincere thanks to the contributors.
