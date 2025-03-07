# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from .vit import VisionTransformer,Mlp

from semilearn.nets.utils import load_checkpoint
from .util.pos_embed import get_2d_sincos_pos_embed

class MaskedAutoencoderViT(VisionTransformer):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, mlp_ratio=4., in_chans=3,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 norm_pix_loss=False, use_cls=True, **kwargs):
        

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        super().__init__(**kwargs) # inherit from Vanila ViT
        # self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        # num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        # self.blocks = nn.ModuleList([
        #     Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(depth)])
        # self.norm = norm_layer(embed_dim)
        # # --------------------------------------------------------------------------
        
        # Classifer specifics
        if use_cls:
            self.mae_encoder_head =Mlp(in_features=self.embed_dim, 
                                    hidden_features=self.embed_dim * 4, 
                                    out_features= self.num_classes, 
                                    act_layer=nn.GELU)
            # self.mae_head_norm = nn.LayerNorm(self.embed_dim)
        # self.head = Mlp(in_features=self.embed_dim, 
        #                 hidden_features=self.embed_dim * 4, 
        #                 out_features= self.num_classes, 
        #                 act_layer=nn.GELU)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        print("kwargs in MaskedAutoencoderViT", kwargs)
        print("decoder depth: {}; dim:{}; head {}".format(decoder_depth, decoder_embed_dim, decoder_num_heads))
        self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        num_patches = self.patch_embed.num_patches
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        norm_layer = nn.LayerNorm
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
    
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, kwargs['patch_size']**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :].detach()

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :].detach()
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_maeloss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        return loss and mask [N, L] loss per patch
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = loss * mask # mean loss on removed patches
        return loss, mask
    

    def forward_mae(self, imgs, mask_ratio=0.5, mask_avg_pool=False):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        # print("latent size", latent.size())
        
        # classifier from masked latent
        if mask_avg_pool:
            x = latent[:, 1:].mean(dim=1)
        else:
            x = latent[:, 0]

        mae_global_pool_logic = self.mae_encoder_head(x)

        # print("mae_global_pool_logic size ", mae_global_pool_logic.size())
        # mae_global_pool_logic = self.mae_head_norm(mae_global_pool_logic)

        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss, mask = self.forward_maeloss(imgs, pred, mask)

        return loss, pred, mask, mae_global_pool_logic
    

    def forward(self, x, only_fc=False, only_feat=False, need_mae = False, mask_avg_pool=False, mask_ratio=0.5, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """
        
        if only_fc:
            return self.head(x)
        
        feateal_x = self.extract(x)
        if self.global_pool:
            feateal_x = feateal_x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else feateal_x[:, 0]
        feateal_x = self.fc_norm(feateal_x)

        if only_feat:
            return feateal_x

        output = self.head(feateal_x)

        # add mae part
        if need_mae:
            mae_loss, mae_pred, mae_mask, mae_global_pool_logic= self.forward_mae(x, mask_avg_pool=mask_avg_pool, mask_ratio=mask_ratio)
        else:
            mae_loss = mae_mask = None
            mae_global_pool_logic = None

        result_dict = {'logits':output, 'feat':feateal_x,
                       'mae_logits': mae_global_pool_logic, 
                       'mae_loss':mae_loss,
                       'mae_mask':mae_mask,
                       }
        return result_dict

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

def mae_vit_tiny_patch2_32(pretrained=False, pretrained_path=None, **kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    model_kwargs = dict(img_size=32, patch_size=2, embed_dim=192, depth=12, num_heads=3, drop_path_rate=0.1, decoder_embed_dim=192, decoder_depth=12, decoder_num_heads=3, **kwargs)
    model = MaskedAutoencoderViT(**model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model


def mae_vit_small_patch2_32(pretrained=False, pretrained_path=None, decoder_depth=4, decoder_width= 192, **kwargs):
    """ ViT-Small (ViT-S/2)
    """
    model_kwargs = dict(img_size=32, patch_size=2, embed_dim=384, depth=12, num_heads=6, drop_path_rate=0.2, decoder_embed_dim=decoder_width, decoder_depth=decoder_depth, decoder_num_heads=3, **kwargs)
    model = MaskedAutoencoderViT(**model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model

def mae_vit_small_patch16_224(pretrained=False, pretrained_path=None, **kwargs):
    """ ViT-Small (ViT-S/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, drop_path_rate=0.1, 
    decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=3, norm_pix_loss=False, **kwargs)
    model = MaskedAutoencoderViT(**model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)    
    return model


def mae_vit_base_patch16_96(pretrained=False, pretrained_path=None, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(img_size=96, patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.2, 
    decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=6, **kwargs)
    model = MaskedAutoencoderViT(**model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)   
    return model


def mae_vit_base_patch16_224(pretrained=False, pretrained_path=None, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.2,
    decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=6, **kwargs)
    model = MaskedAutoencoderViT(**model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)   
    return model

if __name__=='__main__':
    model = mae_vit_tiny_patch2_32()
    print(model)
