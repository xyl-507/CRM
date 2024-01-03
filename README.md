# CRM

This project hosts the code for implementing the CRM module for visual tracking, as presented in our paper: 

```
Consistent Representation Mining for Multi-Drone Single Object Tracking
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


[**[Models(baidu:iiau) and Raw Results]**](https://pan.baidu.com/s/15ntlgipFTmzKDclilrEg1A?pwd=1234)


### Multi-UAV Tracking

| Model(arch+backbone)  | MDOT (Suc./Pre.)   | Drone1 (Suc./Pre.)| Drone2 (Suc./Pre.) |
| --------------------  | :----------------: | :---------------: | :---------------:  |
| SiamBAN(baseline)     |    0.394/0.570     |    0.426/0.619    |    0.363/0.521     |
| CRM-Siam              |    0.401/0.602     |    0.417/0.627    |    0.385/0.577     |
| Super-DiMP(baseline)  |    0.506/0.728     |    0.536/0.769    |    0.476/0.687     |
| CRM-DiMP              |    0.524/0.759     |    0.535/0.774    |    0.513/0.743     |


### Acknowledgement
The code based on the [PyTracking](https://github.com/visionml/pytracking), [PySOT](https://github.com/STVIR/pysot) , [SiamBAN](https://github.com/hqucv/siamban) ,
[Biformer](https://ieeexplore.ieee.org/document/10203555).
We would like to express our sincere thanks to the contributors.
