'''
Attribute wise evaluation of datasets
https://github.com/visionml/pytracking/issues/323
'''
import os
import sys
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [14, 8]

sys.path.append('../..')
from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results, \
    print_results_per_attribute, plot_attributes_radar
from pytracking.evaluation import Tracker, get_dataset, trackerlist, get_dataset_attributes
# 待评估的跟踪器
trackers = []
trackers.extend(trackerlist('atom', 'default', range(0, 1), 'ATOM'))  # range(0,5) 代表是测五个算法
# trackers.extend(trackerlist('dimp', 'dimp18', range(0,5), 'DiMP18'))
# trackers.extend(trackerlist('dimp', 'dimp50', range(0,5), 'DiMP50'))
# trackers.extend(trackerlist('dimp', 'prdimp18', range(0,5), 'PrDiMP18'))
# trackers.extend(trackerlist('dimp', 'prdimp50', range(0,5), 'PrDiMP50'))
# 待评估的数据集
dataset = get_dataset('uav')
# dataset = get_dataset('uav', 'dtb', 'uavdt')

# ----------------------------------------------------------- Plots for UAV， 画成功率和准确率的图
# 图片保存在 /home/xyl/xyl-code/pytracking-master/pytracking/result_plots
plot_results(trackers, dataset, 'UAV', merge_results=True, plot_types=('success', 'prec'),
             skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

# ----------------------------------------------------------- Tables for UAV 打印数据集的总的结果
print_results(trackers, dataset, 'UAV', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))

# ----------------------------------------------------------- Print per sequence results for all sequences 打印每个序列上的结果
# filter_criteria = None
# dataset = get_dataset('uav')
# print_per_sequence_results(trackers, dataset, 'UAV', merge_results=True,
#                            filter_criteria=filter_criteria, force_evaluation=False)

# ----------------------------------------------------------- print_results_per_attribute 打印每个属性上的结果， mode 是选择属性的缩写与否
# print_results_per_attribute(trackers, get_dataset_attributes('uav', mode='short'), 'UAV',
#                             merge_results=True, force_evaluation=False,
#                             skip_missing_seq=True,
#                             exclude_invalid_frames=False)

# ----------------------------------------------------------- plot_attributes_radar  画雷达图
plot_attributes_radar(trackers,
                      get_dataset_attributes('uav', mode='long'), 'UAV',
                      merge_results=True, force_evaluation=False,
                      skip_missing_seq=True,
                      plot_opts=None, exclude_invalid_frames=False)

# ----------------------------------------------------------- plot the curves for each attribute，每个属性一个文件夹
# for key, dataset in get_dataset_attributes('uav', mode='short').items():
#     print(key)
#     plot_results(trackers, dataset, 'uav' + key, merge_results=True, force_evaluation=False, skip_missing_seq=False,
#                  plot_types=('success'))



'''
pytracking的analyze_results.ipynb
'''
# dataset = get_dataset('otb', 'nfs', 'uav')
# plot_results(trackers, dataset, 'OTB+NFS+UAV', merge_results=True, plot_types=('success', 'prec'), 
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)


# # ## Plots for LaSOT
#
# # In[ ]:
#
#
# trackers = []
# trackers.extend(trackerlist('atom', 'default', range(0,5), 'ATOM'))
# trackers.extend(trackerlist('dimp', 'dimp18', range(0,5), 'DiMP18'))
# trackers.extend(trackerlist('dimp', 'dimp50', range(0,5), 'DiMP50'))
# trackers.extend(trackerlist('dimp', 'prdimp18', range(0,5), 'PrDiMP18'))
# trackers.extend(trackerlist('dimp', 'prdimp50', range(0,5), 'PrDiMP50'))
#
# dataset = get_dataset('lasot')
# plot_results(trackers, dataset, 'LaSOT', merge_results=True, plot_types=('success'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
#
#
# # ## Tables for OTB, NFS, UAV and LaSOT
#
# # In[ ]:
#
#
# trackers = []
# trackers.extend(trackerlist('atom', 'default', range(0, 1), 'ATOM'))
# trackers.extend(trackerlist('dimp', 'dimp18', range(0,5), 'DiMP18'))
# trackers.extend(trackerlist('dimp', 'dimp50', range(0,5), 'DiMP50'))
# trackers.extend(trackerlist('dimp', 'prdimp18', range(0,5), 'PrDiMP18'))
# trackers.extend(trackerlist('dimp', 'prdimp50', range(0,5), 'PrDiMP50'))
#
# dataset = get_dataset('otb')
# print_results(trackers, dataset, 'OTB', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
#
# dataset = get_dataset('nfs')
# print_results(trackers, dataset, 'NFS', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
#
# dataset = get_dataset('uav')
# print_results(trackers, dataset, 'UAV', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
#
# dataset = get_dataset('otb', 'nfs', 'uav')
# print_results(trackers, dataset, 'OTB+NFS+UAV', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
#
# dataset = get_dataset('lasot')
# print_results(trackers, dataset, 'LaSOT', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
#
#
# # ## Filtered per-sequence results
#
# # In[ ]:
#
#
# # Print per sequence results for sequences where all trackers fail, i.e. all trackers have average overlap in percentage of less than 10.0
# filter_criteria = {'mode': 'ao_max', 'threshold': 10.0}
# dataset = get_dataset('otb', 'nfs', 'uav')
# print_per_sequence_results(trackers, dataset, 'OTB+NFS+UAV', merge_results=True, filter_criteria=filter_criteria, force_evaluation=False)
#
#
# # In[ ]:
#
#
# # Print per sequence results for sequences where at least one tracker fails, i.e. a tracker has average overlap in percentage of less than 10.0
# filter_criteria = {'mode': 'ao_min', 'threshold': 10.0}
# dataset = get_dataset('otb', 'nfs', 'uav')
# print_per_sequence_results(trackers, dataset, 'OTB+NFS+UAV', merge_results=True, filter_criteria=filter_criteria, force_evaluation=False)
#
#
# # In[ ]:
#
#
# # Print per sequence results for sequences where the trackers have differing behavior.
# # i.e. average overlap in percentage for different trackers on a sequence differ by at least 40.0
# filter_criteria = {'mode': 'delta_ao', 'threshold': 40.0}
# dataset = get_dataset('otb', 'nfs', 'uav')
# print_per_sequence_results(trackers, dataset, 'OTB+NFS+UAV', merge_results=True, filter_criteria=filter_criteria, force_evaluation=False)
#
#
# # In[ ]:
#
#
# # Print per sequence results for all sequences
# filter_criteria = None
# dataset = get_dataset('otb', 'nfs', 'uav')
# print_per_sequence_results(trackers, dataset, 'OTB+NFS+UAV', merge_results=True, filter_criteria=filter_criteria, force_evaluation=False)
#
