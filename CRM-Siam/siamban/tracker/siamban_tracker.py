from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from siamban.core.config import cfg
from siamban.tracker.base_tracker import SiameseTracker
from siamban.utils.bbox import corner2center


class SiamBANTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamBANTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
                          cfg.POINT.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = cfg.BAN.KWARGS.cls_out_channels
        self.window = window.flatten()
        self.points = self.generate_points(cfg.POINT.STRIDE, self.score_size)
        self.model = model
        self.model.eval()
        self.theta = 0.6  # xyl

    def generate_points(self, stride, size):
        ori = - (size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    def _convert_bbox(self, delta, point):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()

        delta[0, :] = point[:, 0] - delta[0, :]
        delta[1, :] = point[:, 1] - delta[1, :]
        delta[2, :] = point[:, 0] + delta[2, :]
        delta[3, :] = point[:, 1] + delta[3, :]
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
            score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def init2(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos2 = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                     bbox[1] + (bbox[3] - 1) / 2])
        self.size2 = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size2[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size2)
        h_z = self.size2[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size2)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average2 = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos2,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average2)
        self.model.template2(z_crop)

    def track(self, img, idx, v_idx):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)

        # score = self._convert_score(outputs['cls'])
        # pred_bbox = self._convert_bbox(outputs['loc'], self.points)

        score1 = self._convert_score(outputs['cls1'])
        pred_bbox1 = self._convert_bbox(outputs['loc1'], self.points)
        score2 = self._convert_score(outputs['cls2'])
        pred_bbox2 = self._convert_bbox(outputs['loc2'], self.points)
        # 判断条件 - UAV1
        if max(score1) < self.theta < max(score2):
            score = score2
            pred_bbox = pred_bbox2
        else:
            score = score1
            pred_bbox = pred_bbox1
            self.counting1 = 1

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
            'bbox': bbox,
            'best_score': best_score,
        }

    def track2(self, img, idx, v_idx):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size2[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size2)
        h_z = self.size2[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size2)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos2,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average2)

        outputs = self.model.track2(x_crop)

        # score = self._convert_score(outputs['cls'])
        # pred_bbox = self._convert_bbox(outputs['loc'], self.points)

        score1 = self._convert_score(outputs['cls1'])
        pred_bbox1 = self._convert_bbox(outputs['loc1'], self.points)
        score2 = self._convert_score(outputs['cls2'])
        pred_bbox2 = self._convert_bbox(outputs['loc2'], self.points)
        # 判断条件 - UAV2
        if max(score1) > self.theta > max(score2):
            score = score1
            pred_bbox = pred_bbox1
        else:
            score = score2
            pred_bbox = pred_bbox2

        # ------------------------------------------------------------------------------------------
        # import cv2
        # import os
        # pred_score = outputs['cls1']
        # f = pred_score.squeeze().cpu().detach().numpy()  # f.shape:  (2, 25, 25)
        # f = f.transpose(1, 2, 0)[:, :, 1]  # [:,:,0/1] # f.shape:  (25, 25)
        #
        # # show_tensor_np(f, i, 'heatmap')  # 蓝色的分类图-DROL
        #
        # # print('img.shape: ', img.shape)  # img.shape:  (720, 1280, 3)
        # # img1 = img  # 用 x_crop
        # img1 = x_crop.squeeze()  # 用 x_crop作为原图
        # img1 = img1.permute(1, 2, 0)  # 120 或者 210图像是反的  # img1.shape:  torch.Size([255, 255, 3])
        # # print('img1.shape: ', img1.shape)  # 三维的transpose只能有两个参数
        # a = np.maximum(f, 0)  # a.shape:  (25, 25)
        # a /= np.max(a)  # 归一化
        # heatmap = cv2.resize(a, (img1.shape[1], img1.shape[0]))  # img.shape->(255,255,3)
        # heatmap = np.uint8(255 * heatmap)
        #
        # # heatmap = 255 - heatmap  # 红蓝颜色反转
        #
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # heatmap.shape:  (255, 255, 3)
        # img1 = img1.cpu().detach().numpy()
        # heatmap_sum = heatmap * 0.5 + img1  # 注释掉就没有原图像进行叠加 # heatmap_sum.shape:  (255, 255, 3)
        # # heatmap_sum = np.ascontiguousarray(heatmap_sum)  # TypeError: Expected Ptr<cv::UMat> for argument 'img'
        # img1 = np.ascontiguousarray(img1)  # TypeError: Expected Ptr<cv::UMat> for argument 'img'
        # heatmap = np.ascontiguousarray(heatmap)  # TypeError: Expected Ptr<cv::UMat> for argument 'img'
        # v_idx += 1
        # save_path = os.path.join('/home/xyl/xyl-code/Multi-SmallTrack/demo/response/heatmap', str(v_idx))
        # if not os.path.isdir(save_path):
        #     os.makedirs(save_path)
        # save_path = os.path.join('/home/xyl/xyl-code/Multi-SmallTrack/demo/response/img_orign2', str(v_idx))
        # if not os.path.isdir(save_path):
        #     os.makedirs(save_path)
        # save_path = os.path.join('/home/xyl/xyl-code/Multi-SmallTrack/demo/response/heatmap+img', str(v_idx))
        # if not os.path.isdir(save_path):
        #     os.makedirs(save_path)
        # cv2.imwrite('/home/xyl/xyl-code/Multi-SmallTrack/demo/response/heatmap/{}/heatmap_{:06d}.jpg'.format(v_idx, idx), heatmap)
        # cv2.imwrite('/home/xyl/xyl-code/Multi-SmallTrack/demo/response/img_orign2/{}/img_orign_{:06d}.jpg'.format(v_idx, idx), img1)
        # cv2.imwrite('/home/xyl/xyl-code/Multi-SmallTrack/demo/response/heatmap+img/{}/heatmap+img_{:06d}.jpg'.format(v_idx, idx), heatmap_sum)
        # ------------------------------------------------------------------------------------------
        # pred_score = outputs['cls2']
        # f = pred_score.squeeze().cpu().detach().numpy()  # f.shape:  (2, 25, 25)
        # f = f.transpose(1, 2, 0)[:, :, 1]  # [:,:,0/1] # f.shape:  (25, 25)
        #
        # # show_tensor_np(f, i, 'heatmap')  # 蓝色的分类图-DROL
        #
        # # print('img.shape: ', img.shape)  # img.shape:  (720, 1280, 3)
        # # img1 = img  # 用 x_crop
        # # img1 = x_crop.squeeze()  # 用 x_crop作为原图
        # # img1 = img1.permute(1, 2, 0)  # 120 或者 210图像是反的  # img1.shape:  torch.Size([255, 255, 3])
        # # print('img1.shape: ', img1.shape)  # 三维的transpose只能有两个参数
        # a = np.maximum(f, 0)  # a.shape:  (25, 25)
        # a /= np.max(a)  # 归一化
        # heatmap = cv2.resize(a, (img1.shape[1], img1.shape[0]))  # img.shape->(255,255,3)
        # heatmap = np.uint8(255 * heatmap)
        #
        # # heatmap = 255 - heatmap  # 红蓝颜色反转
        #
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # heatmap.shape:  (255, 255, 3)
        # # img1 = img1.cpu().detach().numpy()
        # heatmap_sum = heatmap * 0.5 + img1  # 注释掉就没有原图像进行叠加 # heatmap_sum.shape:  (255, 255, 3)
        # # heatmap_sum = np.ascontiguousarray(heatmap_sum)  # TypeError: Expected Ptr<cv::UMat> for argument 'img'
        # # img1 = np.ascontiguousarray(img1)  # TypeError: Expected Ptr<cv::UMat> for argument 'img'
        # heatmap = np.ascontiguousarray(heatmap)  # TypeError: Expected Ptr<cv::UMat> for argument 'img'
        # save_path = os.path.join('/home/xyl/xyl-code/Multi-SmallTrack/demo/response/heatmap2', str(v_idx))
        # if not os.path.isdir(save_path):
        #     os.makedirs(save_path)
        # save_path = os.path.join('/home/xyl/xyl-code/Multi-SmallTrack/demo/response/heatmap+img2', str(v_idx))
        # if not os.path.isdir(save_path):
        #     os.makedirs(save_path)
        # cv2.imwrite('/home/xyl/xyl-code/Multi-SmallTrack/demo/response/heatmap2/{}/heatmap_{:06d}.jpg'.format(v_idx, idx), heatmap)
        # # cv2.imwrite('/home/xyl/xyl-code/Multi-SmallTrack/demo/response/img_orign2/img_orign_{:06d}.jpg'.format(idx), img1)
        # cv2.imwrite('/home/xyl/xyl-code/Multi-SmallTrack/demo/response/heatmap+img2/{}/heatmap+img_{:06d}.jpg'.format(v_idx, idx), heatmap_sum)


        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size2[0] * scale_z, self.size2[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size2[0] / self.size2[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos2[0]
        cy = bbox[1] + self.center_pos2[1]

        # smooth bbox
        width = self.size2[0] * (1 - lr) + bbox[2] * lr
        height = self.size2[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos2 = np.array([cx, cy])
        self.size2 = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
            'bbox': bbox,
            'best_score': best_score,
            'counting': self.counting2
        }
