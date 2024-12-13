"""
   Author: Li Guo
   Modified from BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
"""

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange
import copy
import torch.nn.functional as F
import math 



class PartAttention(nn.Module):
    """
    @article{he2021transfg,
    title={TransFG: A Transformer Architecture for Fine-grained Recognition},
    author={He, Ju and Chen, Jie-Neng and Liu, Shuai and Kortylewski, Adam and Yang, Cheng and Bai, Yutong and Wang, Changhu and Yuille, Alan},
    journal={arXiv preprint arXiv:2103.07976},
    year={2021}
    }
    """
    def __init__(self):
        super(PartAttention, self).__init__()

    def forward(self, x, k=6):
        """
        x -> list: the attention list from the encoder
        k: select the top k attention score and their index from each head
        """
        length = len(x)
        last_map = x[0]
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)
        last_map = last_map[:, :, 0, 1:]
        max_attn, max_inx = last_map.sort(2, descending=True)
        max_inx = max_inx[:, :, :k].reshape([last_map.size(0), -1])
        max_attn = max_attn[:, :, :k].reshape([last_map.size(0), -1])
        return max_attn, max_inx

class LocalSample(nn.Module):
    def __init__(self):
        super(LocalSample, self).__init__()
        self.part_select = PartAttention()
        # self.causal_layer = CaaM(embed_dim, num_heads, ff_dim, dropout)

    def forward(self, x, attn, k):
        """
        x: all visual tokens
        attn: attention from encoder
        k: select k local feature

        fl: k local feature without [CLS] token
        """
        part_attn, part_inx = self.part_select(attn, k)
        part_inx = part_inx + 1
        parts = []
        B, num = part_inx.shape
        for i in range(B):
            parts.append(x[i, part_inx[i, :]])
        fl = torch.stack(parts).squeeze(1)
        fl = torch.cat([x[:, :1], fl], dim=1)
        # fl = self.causal_layer(fl)[:, 1:]
        return fl
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)  
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias) 
        
        # Normalization layers
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x, context):
        """
        Args:
            x: query input [B, N_q, C]  # current image features
            context: key/value input [B, N_kv, C]  # cluster centers of the dataset
        """
        B, N_q, C = x.shape # (B, num_patches, dim)
        context = context.unsqueeze(0).expand(B, -1, -1)  # [512, 768] -> [B, 512, 768]
        N_kv = context.shape[1]  # context (512, 768)
        
        # Linear projections and reshape
        q = self.q(x).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N_q, head_dim]
        kv = self.kv(context).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # k, v: [B, num_heads, N_kv, head_dim]

        # Apply normalization and scaling
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale

        #  Attention
        attn = q @ k.transpose(-2, -1)  # [B, num_heads, N_q, N_kv]
        attn = attn.softmax(dim=-1)
        
        # Combine values
        x = attn @ v  # [B, num_heads, N_q, head_dim]

        # Reshape and project output
        x = x.transpose(1, 2).reshape(B, N_q, C)  # [B, N_q, C]
        x = self.proj(x)
        return x

class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih * self.iw, w=self.ih * self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class DownSamplingTrans(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x
    
class GlobalSample(nn.Module):
    def __init__(self, embed_dim):
        super(GlobalSample, self).__init__()
        self.global_sample = DownSamplingTrans(embed_dim, embed_dim, (7, 7), downsample=True)

    def forward(self, x):
        img = x[:, 1:, :].contiguous()
        B, L, N = img.size()
        img = img.reshape([-1, 14, 14, N]).contiguous()

        img = img.permute(0, 3, 1, 2).contiguous()
        img = self.global_sample(img)
        img = img.permute(0, 2, 3, 1).contiguous()

        fg = img.reshape([B, -1, N]).contiguous()
        return fg


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous()
        x =x.view(nbatches, -1, self.h * self.d_k).contiguous()
        return self.linears[-1](x)

class LGFM(nn.Module):
    """
    Local-Global Fuse Module
    """
    def __init__(self, embed_dim):
        super(LGFM, self).__init__()
        self.embed_dim = embed_dim
        self.norm_l = LayerNorm(embed_dim)
        self.norm_g = LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        self.llf = MultiHeadedAttention(8, embed_dim)
        self.lgf = MultiHeadedAttention(8, embed_dim)
        self.proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = LayerNorm(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, fl, fg):
        fl = self.norm_l(fl)
        fg = self.norm_g(fg)
        fll = fl + self.llf(fl, fl, fl)
        flg = fl + self.lgf(fl, fg, fg)
        out = self.proj(torch.cat([fll, flg], dim=-1))
        out = self.norm(out)
        out = out + self.fc(out)
        return out

class FDIntervention(nn.Module):
    """
    Front-Door Intervention Module
    """
    def __init__(self, embed_dim):
        super(FDIntervention, self).__init__()
        self.embed_dim = embed_dim
        self.af_1 = AF(embed_dim)

    def forward(self, feature, mediator, proj=False):
        out = self.af_1(feature, mediator, mediator, proj)
        return out

class AF(nn.Module):
    """
    Attention Fuse Module
    """
    def __init__(self, embed_dim):
        super(AF, self).__init__()
        self.embed_dim = embed_dim
        self.q = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, proj=False):
        if proj:
            q = self.q(q)
            qk = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.embed_dim)
            score = qk.softmax(dim=-1)
            out = score.matmul(v)
            out = self.out(out)
        else:
            qk = torch.matmul(q, k.transpose(-1, -2))
            score = qk.softmax(dim=-1)
            out = score.matmul(v)
        return out


class VDM(nn.Module):
    def __init__(self, embed_dim):
        super(VDM, self).__init__()
        self.fuse = LGFM(embed_dim)
        self.intervene = FDIntervention(embed_dim)

    def forward(self, x, img_local =None, img_global=None, proj=False):
        """
        fl: local feature
        fg: global feature
        """
        x = x.contiguous()
        if img_local is not None:
            img_local = img_local.contiguous()
        if img_global is not None:
            img_global = img_global.contiguous()
            
        mt = self.fuse(img_local, img_global)
        mt = mt.contiguous()
        
        out = self.intervene(x, mt, proj)
        out = out.contiguous()

        return (out).contiguous()