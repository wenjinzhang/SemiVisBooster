# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.models.layers import DropPath, trunc_normal_
from timm.models.layers.helpers import to_2tuple

from semilearn.nets.utils import load_checkpoint


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class VisionProjection(nn.Module):
    def __init__(self, input_dim=384, output_dim=512):
        super(VisionProjection, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),  # Add non-linearity
            nn.BatchNorm1d(output_dim),  # Optional: normalize the output
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, vision_embeds):
        return self.proj(vision_embeds)


class AttentionProjection(nn.Module):
    def __init__(self, vision_dim, text_dim, hidden_dim=512):
        super(AttentionProjection, self).__init__()
        self.query = nn.Linear(vision_dim, hidden_dim)
        self.key = nn.Linear(text_dim, hidden_dim)
        self.value = nn.Linear(text_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vision_dim)  # Final projection to vision space

    def forward(self, vision_embeds, text_embeds):
        # Compute query, key, and value
        q = self.query(vision_embeds)
        k = self.key(text_embeds)
        v = self.value(text_embeds)

        # Compute attention weights
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention
        attended_embeds = torch.matmul(attention_weights, v)
        projected_embeds = self.fc(attended_embeds)

        return projected_embeds


class AttentionProjector(nn.Module):
    def __init__(self, vision_dim, text_dim, hidden_dim=512):
        super(AttentionProjector, self).__init__()
        self.query = nn.Linear(vision_dim, hidden_dim)  # Vision to hidden_dim
        self.key = nn.Linear(text_dim, hidden_dim)  # Text to hidden_dim
        self.value = nn.Linear(text_dim, hidden_dim)  # Text to hidden_dim
        self.fc = nn.Linear(hidden_dim, text_dim)  # Output back to text_dim

    def forward(self, vision_embeds, text_embeds):
        # Compute query (from vision), key and value (from text)
        q = self.query(vision_embeds)  # Map vision embeddings to hidden_dim (query)
        k = self.key(text_embeds)  # Map text embeddings to hidden_dim (key)
        v = self.value(text_embeds)  # Map text embeddings to hidden_dim (value)

        # Compute attention weights
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)  # Scaled dot-product attention
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention weights to the value
        attended_embeds = torch.matmul(attention_weights, v)

        # Project the attended embeddings to the text embedding dimension
        projected_embeds = self.fc(attended_embeds)
        return projected_embeds


class MultiHeadAttentionProjector(nn.Module):
    def __init__(self, vision_dim, text_dim, hidden_dim=512, num_heads=4):
        super(MultiHeadAttentionProjector, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear transformations for query, key, value for each attention head
        self.query = nn.Linear(vision_dim, hidden_dim)
        self.key = nn.Linear(text_dim, hidden_dim)
        self.value = nn.Linear(text_dim, hidden_dim)
        
        # Output linear transformation
        self.fc_out = nn.Linear(hidden_dim, text_dim)

    def forward(self, vision_embeds, text_embeds):
        # Compute query, key, and value and split across heads
        batch_size = vision_embeds.size(0)
        
        q = self.query(vision_embeds).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.key(text_embeds).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.value(text_embeds).view(batch_size, -1, self.num_heads, self.head_dim)

        # Transpose to get shape (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention for each head
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention weights to the value
        attended_embeds = torch.matmul(attention_weights, v)

        # Concatenate attention outputs from all heads
        attended_embeds = attended_embeds.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # Final linear transformation to text_dim
        projected_embeds = self.fc_out(attended_embeds)
        
        projected_embeds = projected_embeds.squeeze(1)  # Remove the [1] sequence dimension

        return projected_embeds


class MultiHeadSelfAttentionProjector(nn.Module):
    def __init__(self, text_dim, vision_dim, hidden_dim=512, num_heads=8):
        """
        Multi-head self-attention projector that maps text embeddings to the vision embedding dimension.

        Args:
        - text_dim: The dimensionality of the input text embeddings.
        - vision_dim: The dimensionality of the output (i.e., visual embedding space).
        - hidden_dim: The hidden dimension for the attention mechanism (default is 512).
        - num_heads: The number of attention heads (default is 8).
        """
        super(MultiHeadSelfAttentionProjector, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear transformations for query, key, and value
        self.query = nn.Linear(text_dim, hidden_dim)
        self.key = nn.Linear(text_dim, hidden_dim)
        self.value = nn.Linear(text_dim, hidden_dim)

        # Output projection to match the vision embedding dimension
        self.fc_out = nn.Linear(hidden_dim, vision_dim)

    def forward(self, text_embeds):
        batch_size = text_embeds.size(0)

        # Compute query, key, and value, and split into multiple heads
        q = self.query(text_embeds).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.key(text_embeds).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.value(text_embeds).view(batch_size, -1, self.num_heads, self.head_dim)

        # Transpose for attention calculation: (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Compute the weighted value for each head
        attended_embeds = torch.matmul(attention_weights, v)
        # Concatenate the heads back together: (batch_size, seq_len, num_heads * head_dim)
        attended_embeds = attended_embeds.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # Final linear projection to match the vision embedding dimension
        projected_text_embeds = self.fc_out(attended_embeds)
        projected_text_embeds = projected_text_embeds.squeeze(1)  # Remove the [1] sequence dimension

        return projected_text_embeds
    

class MultiHeadCrossAttentionProjector(nn.Module):
    def __init__(self, text_dim, vision_dim, hidden_dim=512, num_heads=8):
        super(MultiHeadCrossAttentionProjector, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear transformations for query, key, and value
        self.query = nn.Linear(text_dim, hidden_dim)
        self.key = nn.Linear(vision_dim, hidden_dim)
        self.value = nn.Linear(vision_dim, hidden_dim)
        
        # Output linear transformation
        self.fc_out = nn.Linear(hidden_dim, vision_dim)

    def forward(self, vision_embeds, text_embeds):
        # Clone inputs if needed to avoid in-place modification issues
        text_embeds = text_embeds.clone()  # Prevent modifying input
        vision_embeds = vision_embeds.clone()  # Prevent modifying input

        # Compute query, key, and value and split across heads
        batch_size = text_embeds.size(0)
        
        q = self.query(text_embeds).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.key(vision_embeds).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.value(vision_embeds).view(batch_size, -1, self.num_heads, self.head_dim)

        # Transpose to get shape (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention for each head
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention weights to the value
        attended_embeds = torch.matmul(attention_weights, v)

        # Concatenate attention outputs from all heads
        attended_embeds = attended_embeds.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # Final linear transformation to match vision_dim
        projected_embeds = self.fc_out(attended_embeds)
        
        # Ensure no in-place squeeze
        projected_embeds = projected_embeds.squeeze(1).clone()  # Remove sequence dimension and clone to avoid in-place modification issues

        return projected_embeds



class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, text_embed_dim=512, vt_proj='linear'):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init: (str): weight init scheme
            init_values: (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.text_embed_dim = text_embed_dim
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        use_fc_norm = self.global_pool == 'avg'
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.num_features = self.embed_dim
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.vt_proj = vt_proj
        if self.vt_proj == 'non-linear':
            print("using non_linear_proj--------------------------")
            self.project_layer = VisionProjection(self.embed_dim, self.text_embed_dim)
        elif self.vt_proj == 'linear':
            print("using linear_proj -----------------------------")
            self.project_layer = nn.Linear(self.embed_dim, self.text_embed_dim)
        elif self.vt_proj == 'attn':
            print("using attention_proj -----------------------------")
            self.project_layer = AttentionProjector(self.embed_dim, self.text_embed_dim)
        elif self.vt_proj == 'multi_head-attn':
            print("using multi-head attention projection----------------------------- ")
            self.project_layer = MultiHeadAttentionProjector(self.embed_dim, self.text_embed_dim)
        elif self.vt_proj == 'multi_head_attn_text':
            print("using multi-head attention projection: project text to vision space!!!!!! ")
            self.project_layer = MultiHeadSelfAttentionProjector(self.text_embed_dim, self.embed_dim)
        elif self.vt_proj == 'mlp_text':
            print("using MLP projector: project text to vision space!!!!!! ")
            self.project_layer = nn.Linear(self.text_embed_dim, self.embed_dim)
            
        elif self.vt_proj == 'multi_head_atten_cross_text':
            print("using multi-head <cross attention>: project text to vision space!!!!!! ")
            # def init (self, text_dim, vision_dim, hidden_dim=512, num_heads=8, num_classes=200, ema_decay=0.99)
            self.project_layer = MultiHeadCrossAttentionProjector(
                text_dim=self.text_embed_dim,
                vision_dim=self.embed_dim)
        else:
            raise Exception('Unknown Projection layer')


    def extract(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x
    
    def projector(self, vision_embs=None, text_embs=None, labels=None):
        if self.vt_proj in ['attn', 'multi_head-attn']:
            return self.project_layer(vision_embs, text_embs)
        elif self.vt_proj in ['multi_head_attn_text']:
            return self.project_layer(text_embs)
        elif self.vt_proj in ['mlp_text']:
            return self.project_layer(text_embs)
        elif self.vt_proj in ['multi_head_atten_cross_text']:
            return self.project_layer(vision_embeds=vision_embs, text_embeds=text_embs)
        else:
            return self.project_layer(vision_embs)


    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """
        
        if only_fc:
            return self.head(x)
        
        x = self.extract(x)
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)

        if only_feat:
            return x

        output = self.head(x)
        result_dict = {'logits':output, 'feat':x}
        return result_dict

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
    def group_matcher(self, coarse=False, prefix=''):
        return dict(
            stem=r'^{}cls_token|{}pos_embed|{}patch_embed'.format(prefix, prefix, prefix),  # stem and embed
            blocks=[(r'^{}blocks\.(\d+)'.format(prefix), None), (r'^{}norm'.format(prefix), (99999,))]
        )

def vit_tiny_patch2_32(pretrained=False, pretrained_path=None, **kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    model_kwargs = dict(img_size=32, patch_size=2, embed_dim=192, depth=12, num_heads=3, drop_path_rate=0.1, **kwargs)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    # return model
    return model


def vit_small_patch2_32(pretrained=False, pretrained_path=None, **kwargs):
    """ ViT-Small (ViT-S/2)
    """
    model_kwargs = dict(img_size=32, patch_size=2, embed_dim=384, depth=12, num_heads=6, drop_path_rate=0.2, **kwargs)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model


def vit_small_patch16_224(pretrained=False, pretrained_path=None, **kwargs):
    """ ViT-Small (ViT-S/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, drop_path_rate=0.2, **kwargs)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)    
    return model


def vit_base_patch16_96(pretrained=False, pretrained_path=None, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(img_size=96, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)   
    return model


def vit_base_patch16_224(pretrained=False, pretrained_path=None, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.2, **kwargs)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)   
    return model