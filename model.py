import torch
import random
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp #,PatchEmbed
from timm.models.layers import DropPath, to_2tuple
from functools import partial
from torch import Tensor
from typing import Optional
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from block.DiT_block import modulate, DiTBlock, DiTBlock_nonclip
# from block.unet2 import UNet as U_Net

#################################################################################
#                  Embedding Layers for Timesteps and Patch                     #
#################################################################################
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
    """
    def __init__(self, img_size=28, patch_size=2, stride=2, in_chans=4, embed_dim=512, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class TimestepEmbed(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



#################################################################################
#                                 Core Diffusion Model                                #
#################################################################################
class FinalLayer(nn.Module):
    """
    The final layer of DiM.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size * 2, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class FinalLayer_nonclip(nn.Module):
    """
    The final layer of DiM.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    """
    Diffusion model.
    """
    def __init__(
        self,
        input_size=28,
        patch_size=2,
        strip_size = 2,
        in_channels=4,
        hidden_size=512,
        depth=16,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.input_size = input_size
        self.x_embedder = PatchEmbed(input_size, patch_size, strip_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbed(hidden_size)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size, 
                num_heads=8,
                ) 
            for i in range(depth) 
        ])
        self.blocks_nonclip = nn.ModuleList([
            DiTBlock_nonclip(
                hidden_size=hidden_size, 
                num_heads=8,
                ) 
            for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.final_layer_nonclip = FinalLayer_nonclip(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize mamba layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in mamba blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y2, w, y=None):
        """
        Forward pass of DiM.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, D) tensor of CT, output by BiomedCLIP's image encoder
        y2: (N, T, D) tensor of CT, output by our pre-train CTencoder 
        w: (N, T, 1) tensor of weight, output by our pre-train CTencoder 
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        # y                                      # (N, D)
        y2 = torch.mean(y2, dim=1)               # (N, D)
        # w                                      # (N, T, 1)

        if y is None:
            c = t + y2
            block_outputs = []
            for i in range(self.depth):
                if i == 0:  
                    x = self.blocks_nonclip[i](x, c, w)
                elif i > self.depth/2: 
                    skip_connection = block_outputs[self.depth-i-1]
                    x = self.blocks_nonclip[i](block_outputs[-1] + skip_connection, c, w)
                else: 
                    x = self.blocks_nonclip[i](block_outputs[-1], c, w)
                block_outputs.append(x)
                x = self.final_layer_nonclip(x, c)

        else:
            c1 = t + y
            c2 = t + y2
            c = torch.cat((c1, c2), dim=1)           # (N, 2*D)
            block_outputs = []
            for i in range(self.depth):
                if i == 0:  
                    x = self.blocks[i](x, c, w)
                elif i > self.depth/2: 
                    skip_connection = block_outputs[self.depth-i-1]
                    x = self.blocks[i](block_outputs[-1] + skip_connection, c, w)
                else: 
                    x = self.blocks[i](block_outputs[-1], c, w)
                block_outputs.append(x)
                x = self.final_layer(x, c) 

        # for block in self.blocks:
        #     x = block(x, c, w)                   # (N, T, D)



        
              # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        return x

    
    def forward_with_cfg(self, x, t, y, y2, w, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, y2, w)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

#################################################################################
#                                   DiM Configs                                 #
#################################################################################

#---------------------------------------------------------------------------------------------------
# code reproduction of DiT,
# from paper 'Scalable Diffusion Models with Transformers'.
def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=512, patch_size=2, strip_size=2, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=512, patch_size=4, strip_size=4, **kwargs)

def DiT_XL_7(**kwargs):  # Important! If the input image dimension is not 224*224, change 7 to 8
    return DiT(depth=28, hidden_size=512, patch_size=7, strip_size=7, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=16, hidden_size=512, patch_size=2, strip_size=2, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=16, hidden_size=512, patch_size=4, strip_size=4, **kwargs)

def DiT_L_7(**kwargs):   # Important! If the input image dimension is not 224*224, change 7 to 8
    return DiT(depth=16, hidden_size=512, patch_size=7, strip_size=7, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=8, hidden_size=512, patch_size=2, strip_size=2, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=8, hidden_size=512, patch_size=4, strip_size=4, **kwargs)

def DiT_B_7(**kwargs):   # Important! If the input image dimension is not 224*224, change 7 to 8
    return DiT(depth=8, hidden_size=512, patch_size=7, strip_size=7, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=4, hidden_size=512, patch_size=2, strip_size=2, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=4, hidden_size=512, patch_size=4, strip_size=4, **kwargs)

def DiT_S_7(**kwargs):   # Important! If the input image dimension is not 224*224, change 7 to 8
    return DiT(depth=4, hidden_size=512, patch_size=7, strip_size=7, **kwargs)

def DiT_SB_2(**kwargs):
    return DiT(depth=7, hidden_size=512, patch_size=2, strip_size=2, **kwargs)

# def UNet_2(**kwargs):
#     return U_Net(n_channels=4, out_channels=8, bilinear=True)

DiT_models = {
    #----------------------code reproduction of DiT---------------------------#
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/7': DiT_XL_7,
    'DiT-L/2' : DiT_L_2,   'DiT-L/4' : DiT_L_4,   'DiT-L/7' : DiT_L_7,
    'DiT-B/2' : DiT_B_2,   'DiT-B/4' : DiT_B_4,   'DiT-B/7' : DiT_B_7,
    'DiT-S/2' : DiT_S_2,   'DiT-S/4' : DiT_S_4,   'DiT-S/7' : DiT_S_7,
    'DiT-SB/2' : DiT_SB_2,
    #----------------------code reproduction of U-Net diffusion---------------------------#
    # 'UNet-2': UNet_2,
}
