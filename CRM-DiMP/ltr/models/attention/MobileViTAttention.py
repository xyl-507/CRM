from torch import nn
import torch
from einops import rearrange
# from ltr.models.attention.Biformer import BiLevelRoutingAttention_nchw
from .Biformer import BiLevelRoutingAttention_nchw
# from tools.plotting import show_tensor, show_tensor_np
import imageio
import numpy as np
import matplotlib.image as mpimg  # 读取图片
import matplotlib.pyplot as plt  # 显示图片
import cv2
from timm.models.layers import DropPath, LayerNorm2d, to_2tuple, trunc_normal_

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.ln(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout):
        super().__init__()
        inner_dim = heads * head_dim
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, head_dim, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        out = x
        for att, ffn in self.layers:
            out = out + att(out)
            out = out + ffn(out)
        return out


class MobileViTAttention(nn.Module):
    def __init__(self, in_channel=3, dim=1024, kernel_size=3, patch_size=7, drop_path=0., mlp_ratio=4):  # dim=512
        super().__init__()
        self.in_channel = in_channel
        self.dim = dim
        self.conv2 = nn.Conv2d(in_channel, dim, kernel_size=1)
        self.conv3 = nn.Conv2d(dim, in_channel, kernel_size=1)
        self.bra = BiLevelRoutingAttention_nchw(dim=dim, num_heads=8, topk=4)  # dim=512, num_heads=8, topk=4

    def forward(self, zf):
        x = zf.clone()  # bs,c,h,w  # for CRM-DiMP
        n, c, h, w = x.shape
        if c == self.dim:
            x = self.bra(x)
        else:
            x = self.conv2(x)
            x = self.bra(x)
            x = self.conv3(x)
        y = x + zf
        return y


def get_gt_mask(featmap,  gt_bboxes):
    featmap_sizes = featmap.size()[-2:]
    featmap_strides = 8
    imit_range = [0, 0, 0, 0, 0]
    with torch.no_grad():
        mask_batch = []
        for batch in range(len(gt_bboxes)):
            mask_level = []
            gt_level = gt_bboxes[batch]  # gt_bboxes: BatchsizexNpointx4coordinate
            h, w = featmap_sizes[0], featmap_sizes[1]
            # mask_per_img = torch.zeros([h, w], dtype=torch.double).cuda()
            mask_per_img = torch.zeros([h, w], dtype=torch.float).cuda()
            a = gt_level.shape[0]
            gt_level_map = gt_level / featmap_strides
            lx = max(int(gt_level_map[0]) - imit_range[0], 0)
            rx = min(int(gt_level_map[2]) + imit_range[0], w)
            ly = max(int(gt_level_map[1]) - imit_range[0], 0)
            ry = min(int(gt_level_map[3]) + imit_range[0], h)
            if (lx == rx) or (ly == ry):
                mask_per_img[ly, lx] += 1
            else:
                mask_per_img[ly:ry, lx:rx] += 1
            # mask_per_img = (mask_per_img > 0).double()
            mask_per_img = (mask_per_img > 0).float()
            mask_level.append(mask_per_img)
            mask_batch.append(mask_level)

            mask_batch_level = []
            for level in range(len(mask_batch[0])):
                tmp = []
                for batch in range(len(mask_batch)):
                    tmp.append(mask_batch[batch][level])
                mask_batch_level.append(torch.stack(tmp, dim=0))

        return mask_batch_level


if __name__ == '__main__':

    # 设置运行的设备
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    # m = MobileViTAttention(in_channel=256)
    # input = torch.randn(1, 256, 7, 7).to(device)
    # input = torch.randn(28, 3, 127, 127).to(device)

    img = imageio.imread(r'D:\XYL\3.Object tracking\3.VisualTracking-Toolkit-master\0.demo\\car.jpg')  # array (255,255,3)
    # img = imageio.imread(r'D:\XYL\3.Object tracking\3.VisualTracking-Toolkit-master\0.ILSVRC2015-val\\ILSVRC2012_val_00000073.JPEG')  # array (255,255,3)
    # img = cv2.resize(img, (255, 255))
    img1 = np.transpose(img, (2, 0, 1))  # array (3, 255, 255)
    img2 = torch.tensor(img1.copy())  # tensor (3, 255, 255)
    input = img2.unsqueeze(dim=0).float().to(device)  # tensor (1, 3, 255, 255)

    m = MobileViTAttention(in_channel=3).to(device)
    # m2 = MobileViTAttention(in_channel=3).to(device)

    output = m([input, input, input])  # tensor to list
    # output2 = m2([output[0],output[0],output[0]])
    # output = m([input, input, input], [input, input, input])
    print(output[0].shape)
