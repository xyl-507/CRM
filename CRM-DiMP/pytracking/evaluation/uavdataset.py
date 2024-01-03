import os
import json
import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text


class UAVDataset(BaseDataset):
    """ UAV123 dataset.

    Publication:
        A Benchmark and Simulator for UAV Tracking.
        Matthias Mueller, Neil Smith and Bernard Ghanem
        ECCV, 2016
        https://ivul.kaust.edu.sa/Documents/Publications/2016/A%20Benchmark%20and%20Simulator%20for%20UAV%20Tracking.pdf

    Download the dataset from https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx
    """
    def __init__(self, attribute=None):
        super().__init__()
        self.base_path = self.env_settings.uav_path
        # self.base_path = '/home/xyl/pysot-master/testing_dataset/UAV123'
        self.sequence_info_list = self._get_sequence_info_list()

        self.att_dict = None

        if attribute is not None:
            self.sequence_info_list = self._filter_sequence_info_list_by_attribute(attribute, self.sequence_info_list)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'uav', ground_truth_rect[init_omit:,:],
                        object_class=sequence_info['object_class'])

    def get_attribute_names(self, mode='short'):
        if self.att_dict is None:
            self.att_dict = self._load_attributes()

        names = self.att_dict['att_name_short'] if mode == 'short' else self.att_dict['att_name_long']
        return names

    def _load_attributes(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'dataset_attribute_specs', 'UAV123_attributes.json'), 'r') as f:
            att_dict = json.load(f)
        return att_dict

    def _filter_sequence_info_list_by_attribute(self, att, seq_list):
        if self.att_dict is None:
            self.att_dict = self._load_attributes()

        if att not in self.att_dict['att_name_short']:
            if att in self.att_dict['att_name_long']:
                att = self.att_dict['att_name_short'][self.att_dict['att_name_long'].index(att)]
            else:
                raise ValueError('\'{}\' attribute invalid.')

        # return [s for s in seq_list if att in self.att_dict[s['name'][4:]]]  # UAV123在pytracking中加了前缀uav_
        return [s for s in seq_list if att in self.att_dict[s['name'][0:]]]

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "bike1", "path": "bike1", "startFrame": 1, "endFrame": 3085, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/bike1.txt",
             "object_class": "vehicle"},
            {"name": "bike2", "path": "bike2", "startFrame": 1, "endFrame": 553, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/bike2.txt",
             "object_class": "vehicle"},
            {"name": "bike3", "path": "bike3", "startFrame": 1, "endFrame": 433, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/bike3.txt",
             "object_class": "vehicle"},
            {"name": "bird1_1", "path": "bird1", "startFrame": 1, "endFrame": 253, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/bird1_1.txt",
             "object_class": "bird"},
            {"name": "bird1_2", "path": "bird1", "startFrame": 775, "endFrame": 1477, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/bird1_2.txt",
             "object_class": "bird"},
            {"name": "bird1_3", "path": "bird1", "startFrame": 1573, "endFrame": 2437, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/bird1_3.txt",
             "object_class": "bird"},
            {"name": "boat1", "path": "boat1", "startFrame": 1, "endFrame": 901, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/boat1.txt",
             "object_class": "vessel"},
            {"name": "boat2", "path": "boat2", "startFrame": 1, "endFrame": 799, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/boat2.txt",
             "object_class": "vessel"},
            {"name": "boat3", "path": "boat3", "startFrame": 1, "endFrame": 901, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/boat3.txt",
             "object_class": "vessel"},
            {"name": "boat4", "path": "boat4", "startFrame": 1, "endFrame": 553, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/boat4.txt",
             "object_class": "vessel"},
            {"name": "boat5", "path": "boat5", "startFrame": 1, "endFrame": 505, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/boat5.txt",
             "object_class": "vessel"},
            {"name": "boat6", "path": "boat6", "startFrame": 1, "endFrame": 805, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/boat6.txt",
             "object_class": "vessel"},
            {"name": "boat7", "path": "boat7", "startFrame": 1, "endFrame": 535, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/boat7.txt",
             "object_class": "vessel"},
            {"name": "boat8", "path": "boat8", "startFrame": 1, "endFrame": 685, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/boat8.txt",
             "object_class": "vessel"},
            {"name": "boat9", "path": "boat9", "startFrame": 1, "endFrame": 1399, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/boat9.txt",
             "object_class": "vessel"},
            {"name": "building1", "path": "building1", "startFrame": 1, "endFrame": 469, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/building1.txt",
             "object_class": "other"},
            {"name": "building2", "path": "building2", "startFrame": 1, "endFrame": 577, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/building2.txt",
             "object_class": "other"},
            {"name": "building3", "path": "building3", "startFrame": 1, "endFrame": 829, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/building3.txt",
             "object_class": "other"},
            {"name": "building4", "path": "building4", "startFrame": 1, "endFrame": 787, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/building4.txt",
             "object_class": "other"},
            {"name": "building5", "path": "building5", "startFrame": 1, "endFrame": 481, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/building5.txt",
             "object_class": "other"},
            {"name": "car1_1", "path": "car1", "startFrame": 1, "endFrame": 751, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car1_1.txt",
             "object_class": "car"},
            {"name": "car1_2", "path": "car1", "startFrame": 751, "endFrame": 1627, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car1_2.txt",
             "object_class": "car"},
            {"name": "car1_3", "path": "car1", "startFrame": 1627, "endFrame": 2629, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car1_3.txt",
             "object_class": "car"},
            {"name": "car10", "path": "car10", "startFrame": 1, "endFrame": 1405, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car10.txt",
             "object_class": "car"},
            {"name": "car11", "path": "car11", "startFrame": 1, "endFrame": 337, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car11.txt",
             "object_class": "car"},
            {"name": "car12", "path": "car12", "startFrame": 1, "endFrame": 499, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car12.txt",
             "object_class": "car"},
            {"name": "car13", "path": "car13", "startFrame": 1, "endFrame": 415, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car13.txt",
             "object_class": "car"},
            {"name": "car14", "path": "car14", "startFrame": 1, "endFrame": 1327, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car14.txt",
             "object_class": "car"},
            {"name": "car15", "path": "car15", "startFrame": 1, "endFrame": 469, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car15.txt",
             "object_class": "car"},
            {"name": "car16_1", "path": "car16", "startFrame": 1, "endFrame": 415, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car16_1.txt",
             "object_class": "car"},
            {"name": "car16_2", "path": "car16", "startFrame": 415, "endFrame": 1993, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car16_2.txt",
             "object_class": "car"},
            {"name": "car17", "path": "car17", "startFrame": 1, "endFrame": 1057, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car17.txt",
             "object_class": "car"},
            {"name": "car18", "path": "car18", "startFrame": 1, "endFrame": 1207, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car18.txt",
             "object_class": "car"},
            {"name": "car1_s", "path": "car1_s", "startFrame": 1, "endFrame": 1475, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car1_s.txt",
             "object_class": "car"},
            {"name": "car2", "path": "car2", "startFrame": 1, "endFrame": 1321, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car2.txt",
             "object_class": "car"},
            {"name": "car2_s", "path": "car2_s", "startFrame": 1, "endFrame": 320, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car2_s.txt",
             "object_class": "car"},
            {"name": "car3", "path": "car3", "startFrame": 1, "endFrame": 1717, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car3.txt",
             "object_class": "car"},
            {"name": "car3_s", "path": "car3_s", "startFrame": 1, "endFrame": 1300, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car3_s.txt",
             "object_class": "car"},
            {"name": "car4", "path": "car4", "startFrame": 1, "endFrame": 1345, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car4.txt",
             "object_class": "car"},
            {"name": "car4_s", "path": "car4_s", "startFrame": 1, "endFrame": 830, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car4_s.txt",
             "object_class": "car"},
            {"name": "car5", "path": "car5", "startFrame": 1, "endFrame": 745, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car5.txt",
             "object_class": "car"},
            {"name": "car6_1", "path": "car6", "startFrame": 1, "endFrame": 487, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car6_1.txt",
             "object_class": "car"},
            {"name": "car6_2", "path": "car6", "startFrame": 487, "endFrame": 1807, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car6_2.txt",
             "object_class": "car"},
            {"name": "car6_3", "path": "car6", "startFrame": 1807, "endFrame": 2953, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car6_3.txt",
             "object_class": "car"},
            {"name": "car6_4", "path": "car6", "startFrame": 2953, "endFrame": 3925, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car6_4.txt",
             "object_class": "car"},
            {"name": "car6_5", "path": "car6", "startFrame": 3925, "endFrame": 4861, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car6_5.txt",
             "object_class": "car"},
            {"name": "car7", "path": "car7", "startFrame": 1, "endFrame": 1033, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car7.txt",
             "object_class": "car"},
            {"name": "car8_1", "path": "car8", "startFrame": 1, "endFrame": 1357, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car8_1.txt",
             "object_class": "car"},
            {"name": "car8_2", "path": "car8", "startFrame": 1357, "endFrame": 2575, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car8_2.txt",
             "object_class": "car"},
            {"name": "car9", "path": "car9", "startFrame": 1, "endFrame": 1879, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/car9.txt",
             "object_class": "car"},
            {"name": "group1_1", "path": "group1", "startFrame": 1, "endFrame": 1333, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/group1_1.txt",
             "object_class": "person"},
            {"name": "group1_2", "path": "group1", "startFrame": 1333, "endFrame": 2515, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/group1_2.txt",
             "object_class": "person"},
            {"name": "group1_3", "path": "group1", "startFrame": 2515, "endFrame": 3925, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/group1_3.txt",
             "object_class": "person"},
            {"name": "group1_4", "path": "group1", "startFrame": 3925, "endFrame": 4873, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/group1_4.txt",
             "object_class": "person"},
            {"name": "group2_1", "path": "group2", "startFrame": 1, "endFrame": 907, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/group2_1.txt",
             "object_class": "person"},
            {"name": "group2_2", "path": "group2", "startFrame": 907, "endFrame": 1771, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/group2_2.txt",
             "object_class": "person"},
            {"name": "group2_3", "path": "group2", "startFrame": 1771, "endFrame": 2683, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/group2_3.txt",
             "object_class": "person"},
            {"name": "group3_1", "path": "group3", "startFrame": 1, "endFrame": 1567, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/group3_1.txt",
             "object_class": "person"},
            {"name": "group3_2", "path": "group3", "startFrame": 1567, "endFrame": 2827, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/group3_2.txt",
             "object_class": "person"},
            {"name": "group3_3", "path": "group3", "startFrame": 2827, "endFrame": 4369, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/group3_3.txt",
             "object_class": "person"},
            {"name": "group3_4", "path": "group3", "startFrame": 4369, "endFrame": 5527, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/group3_4.txt",
             "object_class": "person"},
            {"name": "person1", "path": "person1", "startFrame": 1, "endFrame": 799, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person1.txt",
             "object_class": "person"},
            {"name": "person10", "path": "person10", "startFrame": 1, "endFrame": 1021, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person10.txt",
             "object_class": "person"},
            {"name": "person11", "path": "person11", "startFrame": 1, "endFrame": 721, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person11.txt",
             "object_class": "person"},
            {"name": "person12_1", "path": "person12", "startFrame": 1, "endFrame": 601, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person12_1.txt",
             "object_class": "person"},
            {"name": "person12_2", "path": "person12", "startFrame": 601, "endFrame": 1621, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person12_2.txt",
             "object_class": "person"},
            {"name": "person13", "path": "person13", "startFrame": 1, "endFrame": 883, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person13.txt",
             "object_class": "person"},
            {"name": "person14_1", "path": "person14", "startFrame": 1, "endFrame": 847, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person14_1.txt",
             "object_class": "person"},
            {"name": "person14_2", "path": "person14", "startFrame": 847, "endFrame": 1813, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person14_2.txt",
             "object_class": "person"},
            {"name": "person14_3", "path": "person14", "startFrame": 1813, "endFrame": 2923,
             "nz": 6, "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person14_3.txt",
             "object_class": "person"},
            {"name": "person15", "path": "person15", "startFrame": 1, "endFrame": 1339, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person15.txt",
             "object_class": "person"},
            {"name": "person16", "path": "person16", "startFrame": 1, "endFrame": 1147, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person16.txt",
             "object_class": "person"},
            {"name": "person17_1", "path": "person17", "startFrame": 1, "endFrame": 1501, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person17_1.txt",
             "object_class": "person"},
            {"name": "person17_2", "path": "person17", "startFrame": 1501, "endFrame": 2347,
             "nz": 6, "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person17_2.txt",
             "object_class": "person"},
            {"name": "person18", "path": "person18", "startFrame": 1, "endFrame": 1393, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person18.txt",
             "object_class": "person"},
            {"name": "person19_1", "path": "person19", "startFrame": 1, "endFrame": 1243, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person19_1.txt",
             "object_class": "person"},
            {"name": "person19_2", "path": "person19", "startFrame": 1243, "endFrame": 2791,
             "nz": 6, "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person19_2.txt",
             "object_class": "person"},
            {"name": "person19_3", "path": "person19", "startFrame": 2791, "endFrame": 4357,
             "nz": 6, "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person19_3.txt",
             "object_class": "person"},
            {"name": "person1_s", "path": "person1_s", "startFrame": 1, "endFrame": 1600, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person1_s.txt",
             "object_class": "person"},
            {"name": "person2_1", "path": "person2", "startFrame": 1, "endFrame": 1189, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person2_1.txt",
             "object_class": "person"},
            {"name": "person2_2", "path": "person2", "startFrame": 1189, "endFrame": 2623, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person2_2.txt",
             "object_class": "person"},
            {"name": "person20", "path": "person20", "startFrame": 1, "endFrame": 1783, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person20.txt",
             "object_class": "person"},
            {"name": "person21", "path": "person21", "startFrame": 1, "endFrame": 487, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person21.txt",
             "object_class": "person"},
            {"name": "person22", "path": "person22", "startFrame": 1, "endFrame": 199, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person22.txt",
             "object_class": "person"},
            {"name": "person23", "path": "person23", "startFrame": 1, "endFrame": 397, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person23.txt",
             "object_class": "person"},
            {"name": "person2_s", "path": "person2_s", "startFrame": 1, "endFrame": 250, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person2_s.txt",
             "object_class": "person"},
            {"name": "person3", "path": "person3", "startFrame": 1, "endFrame": 643, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person3.txt",
             "object_class": "person"},
            {"name": "person3_s", "path": "person3_s", "startFrame": 1, "endFrame": 505, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person3_s.txt",
             "object_class": "person"},
            {"name": "person4_1", "path": "person4", "startFrame": 1, "endFrame": 1501, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person4_1.txt",
             "object_class": "person"},
            {"name": "person4_2", "path": "person4", "startFrame": 1501, "endFrame": 2743, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person4_2.txt",
             "object_class": "person"},
            {"name": "person5_1", "path": "person5", "startFrame": 1, "endFrame": 877, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person5_1.txt",
             "object_class": "person"},
            {"name": "person5_2", "path": "person5", "startFrame": 877, "endFrame": 2101, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person5_2.txt",
             "object_class": "person"},
            {"name": "person6", "path": "person6", "startFrame": 1, "endFrame": 901, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person6.txt",
             "object_class": "person"},
            {"name": "person7_1", "path": "person7", "startFrame": 1, "endFrame": 1249, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person7_1.txt",
             "object_class": "person"},
            {"name": "person7_2", "path": "person7", "startFrame": 1249, "endFrame": 2065, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person7_2.txt",
             "object_class": "person"},
            {"name": "person8_1", "path": "person8", "startFrame": 1, "endFrame": 1075, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person8_1.txt",
             "object_class": "person"},
            {"name": "person8_2", "path": "person8", "startFrame": 1075, "endFrame": 1525, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person8_2.txt",
             "object_class": "person"},
            {"name": "person9", "path": "person9", "startFrame": 1, "endFrame": 661, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/person9.txt",
             "object_class": "person"},
            {"name": "truck1", "path": "truck1", "startFrame": 1, "endFrame": 463, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/truck1.txt",
             "object_class": "truck"},
            {"name": "truck2", "path": "truck2", "startFrame": 1, "endFrame": 385, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/truck2.txt",
             "object_class": "truck"},
            {"name": "truck3", "path": "truck3", "startFrame": 1, "endFrame": 535, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/truck3.txt",
             "object_class": "truck"},
            {"name": "truck4_1", "path": "truck4", "startFrame": 1, "endFrame": 577, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/truck4_1.txt",
             "object_class": "truck"},
            {"name": "truck4_2", "path": "truck4", "startFrame": 577, "endFrame": 1261, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/truck4_2.txt",
             "object_class": "truck"},
            {"name": "uav1_1", "path": "uav1", "startFrame": 1, "endFrame": 1555, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/uav1_1.txt",
             "object_class": "aircraft"},
            {"name": "uav1_2", "path": "uav1", "startFrame": 1555, "endFrame": 2377, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/uav1_2.txt",
             "object_class": "aircraft"},
            {"name": "uav1_3", "path": "uav1", "startFrame": 2473, "endFrame": 3469, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/uav1_3.txt",
             "object_class": "aircraft"},
            {"name": "uav2", "path": "uav2", "startFrame": 1, "endFrame": 133, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/uav2.txt",
             "object_class": "aircraft"},
            {"name": "uav3", "path": "uav3", "startFrame": 1, "endFrame": 265, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/uav3.txt",
             "object_class": "aircraft"},
            {"name": "uav4", "path": "uav4", "startFrame": 1, "endFrame": 157, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/uav4.txt",
             "object_class": "aircraft"},
            {"name": "uav5", "path": "uav5", "startFrame": 1, "endFrame": 139, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/uav5.txt",
             "object_class": "aircraft"},
            {"name": "uav6", "path": "uav6", "startFrame": 1, "endFrame": 109, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/uav6.txt",
             "object_class": "aircraft"},
            {"name": "uav7", "path": "uav7", "startFrame": 1, "endFrame": 373, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/uav7.txt",
             "object_class": "aircraft"},
            {"name": "uav8", "path": "uav8", "startFrame": 1, "endFrame": 301, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/uav8.txt",
             "object_class": "aircraft"},
            {"name": "wakeboard1", "path": "wakeboard1", "startFrame": 1, "endFrame": 421, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/wakeboard1.txt",
             "object_class": "person"},
            {"name": "wakeboard10", "path": "wakeboard10", "startFrame": 1, "endFrame": 469,
             "nz": 6, "ext": "jpg",
             "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/wakeboard10.txt",
             "object_class": "person"},
            {"name": "wakeboard2", "path": "wakeboard2", "startFrame": 1, "endFrame": 733, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/wakeboard2.txt",
             "object_class": "person"},
            {"name": "wakeboard3", "path": "wakeboard3", "startFrame": 1, "endFrame": 823, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/wakeboard3.txt",
             "object_class": "person"},
            {"name": "wakeboard4", "path": "wakeboard4", "startFrame": 1, "endFrame": 697, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/wakeboard4.txt",
             "object_class": "person"},
            {"name": "wakeboard5", "path": "wakeboard5", "startFrame": 1, "endFrame": 1675, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/wakeboard5.txt",
             "object_class": "person"},
            {"name": "wakeboard6", "path": "wakeboard6", "startFrame": 1, "endFrame": 1165, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/wakeboard6.txt",
             "object_class": "person"},
            {"name": "wakeboard7", "path": "wakeboard7", "startFrame": 1, "endFrame": 199, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/wakeboard7.txt",
             "object_class": "person"},
            {"name": "wakeboard8", "path": "wakeboard8", "startFrame": 1, "endFrame": 1543, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/wakeboard8.txt",
             "object_class": "person"},
            {"name": "wakeboard9", "path": "wakeboard9", "startFrame": 1, "endFrame": 355, "nz": 6,
             "ext": "jpg", "anno_path": "../UAV123-zip/Dataset_UAV123/UAV123/anno/UAV123-orign/wakeboard9.txt",
             "object_class": "person"}
        ]

        return sequence_info_list
