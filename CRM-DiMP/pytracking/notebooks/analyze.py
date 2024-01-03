'''
Attribute wise evaluation of datasets
https://github.com/visionml/pytracking/issues/323
'''
from pytracking.analysis.plot_results import print_results_per_attribute, plot_attributes_radar, plot_results
from pytracking.evaluation import trackerlist , get_dataset_attributes

trackers = []
trackers.extend(trackerlist('dimp', 'super_dimp', range(0,5), 'Super DiMP'))

print_results_per_attribute(trackers, get_dataset_attributes('lasot', mode='short'), 'lasot',
                            merge_results=True, force_evaluation=False,
                            skip_missing_seq=True,
                            exclude_invalid_frames=False)

plot_attributes_radar(trackers,
                      get_dataset_attributes('lasot', mode='long'), 'lasot',
                      merge_results=True, force_evaluation=False,
                      skip_missing_seq=True,
                      plot_opts=None, exclude_invalid_frames=False)
'''
plot the curves for each attribute
'''
trackers = []
trackers.extend(trackerlist('dimp', 'super_dimp', range(0,5), 'Super DiMP'))

for key, dataset in get_dataset_attributes('lasot', mode='short').items():
    print(key)
    plot_results(trackers, dataset, 'lasot' + key, merge_results=True, force_evaluation=False, skip_missing_seq=False, plot_types=('success'))
