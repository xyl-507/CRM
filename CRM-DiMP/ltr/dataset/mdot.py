import torch
import os
import os.path
import numpy as np
import pandas
import random
from collections import OrderedDict
import pdb
from ltr.data.image_loader import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset
from ltr.admin.environment import env_settings


# from ltr.data.dataset_degradation import dataset_degradation
class MDOT(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None):
        self.root = env_settings().mdot_dir if root is None else root
        super().__init__('MDOT', root, image_loader)

        # video_name for each sequence
        self.sequence_list = os.listdir(self.root)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

    def get_name(self):
        return 'mdot'

    def _read_bb_anno(self, seq_path):
        # *****************************************#
        # print("seq_path",seq_path)

        # pdb.set_trace()
        bb_anno_file = os.path.join(seq_path, 'groundtruth.txt')
        #        try:
        #            gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
        #                                 low_memory=False).values
        #        except:
        #            gt = pandas.read_csv(bb_anno_file, delimiter='\t', header=None, dtype=np.float32, na_filter=False,
        #                             low_memory=False).values
        #        else:
        #            gt = pandas.read_csv(bb_anno_file, delimiter=' ', header=None, dtype=np.float32, na_filter=False,
        #                             low_memory=False).values
        try:
            gt = np.loadtxt(bb_anno_file, delimiter=',', dtype=np.float32)
        except:
            gt = np.loadtxt(bb_anno_file, dtype=np.float32)
        return torch.tensor(gt)

    def get_sequence_info(self, seq_name):
        # print("self.root seq_name",self.root,seq_name)
        # pdb.set_trace()
        seq_path = os.path.join(self.root, self.sequence_list[seq_name])
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame(self, seq_path, frame_id):
        frame_path = os.path.join(seq_path, 'img', sorted([p for p in os.listdir(os.path.join(seq_path, 'img')) if
                                                           os.path.splitext(p)[1] in ['.jpg', '.png', '.bmp']])[
            frame_id])
        frame = self.image_loader(frame_path)
        return frame

    def get_frames(self, seq_name, frame_ids, anno=None):
        # print('get_frames'*20)
        # pdb.set_trace()
        seq_path = os.path.join(self.root, self.sequence_list[seq_name])
        # print('seq_path frame_ids', seq_path, frame_ids)
        frame_list = [self._get_frame(seq_path, f) for f in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_path)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        # return frame_list_v, frame_list_i, anno_frames, object_meta
        return frame_list, anno_frames, object_meta
