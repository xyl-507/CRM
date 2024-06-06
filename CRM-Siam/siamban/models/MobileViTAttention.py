from torch import nn
import torch
from einops import rearrange
from siamban.models.Biformer import BiLevelRoutingAttention_nchw
# from siamban.models.vct_help_funcs import TransformerCross


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
    def __init__(self, in_channel=3, dim=512, kernel_size=3, patch_size=7):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size

        self.bra = BiLevelRoutingAttention_nchw(dim=dim, num_heads=8, n_win=7)  # dim=512, num_heads=8, n_win=7

    def forward(self, zf):
        output = []
        for i, x in enumerate(zf):
            y = x.clone()  # bs,c,h,w
            y = self.bra(y)
            y_ = x + y  # residual for Biformer, xyl 20231122
            output.append(y_)
        return output


if __name__ == '__main__':
    m = MobileViTAttention(in_channel=256)
    input = torch.randn(1, 256, 7, 7)
    output = m([input, input, input], [input, input, input])
    print(output.shape)
