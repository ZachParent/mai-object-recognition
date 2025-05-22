from __future__ import absolute_import, division, print_function

import sys
import os
import copy
import logging
import math
from pathlib import Path # Keep for Path operations
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.nn import Conv2d, Dropout, LayerNorm, Linear, Softmax
from torch.nn.modules.utils import _pair

SRC_DIR = Path(__file__).parent.parent
sys.path.append(str(SRC_DIR))

from config import R50_VIT_B16_PRETRAINED_PATH # Adjusted import

# --- Configuration (Using Python Dictionaries) ---
def get_r50_vit_b16_config_dict():
    """Returns the R50+ViT-B/16 configuration as a Python dictionary."""
    config = {}
    config['patches'] = {'size': (16, 16), 'grid': (16, 16)}
    config['hidden_size'] = 768
    config['transformer'] = {
        'mlp_dim': 3072,
        'num_heads': 12,
        'num_layers': 12,
        'attention_dropout_rate': 0.0,
        'dropout_rate': 0.1
    }
    config['resnet'] = {
        'num_layers': (3, 4, 9),
        'width_factor': 1
    }
    config['classifier'] = 'seg'
    # Use the imported constant for the pretrained path
    config['pretrained_path'] = str(R50_VIT_B16_PRETRAINED_PATH)
    
    config['decoder_channels'] = (256, 128, 64, 16)
    config['skip_channels'] = [512, 256, 64, 16]
    config['n_classes'] = 1 
    config['n_skip'] = 3
    config['activation'] = 'sigmoid'
    return config

CONFIGS = {
    'R50-ViT-B_16': get_r50_vit_b16_config_dict(),
}

# ... (logger and NPZ key constants remain the same) ...
logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False): # ... (same)
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x): # ... (same)
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

# --- ResNetV2 Backbone ---
class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)

def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        key_conv1 = f"{n_block}/{n_unit}/conv1/kernel"
        key_conv2 = f"{n_block}/{n_unit}/conv2/kernel"
        key_conv3 = f"{n_block}/{n_unit}/conv3/kernel"
        key_gn1_scale = f"{n_block}/{n_unit}/gn1/scale"
        key_gn1_bias = f"{n_block}/{n_unit}/gn1/bias"
        key_gn2_scale = f"{n_block}/{n_unit}/gn2/scale"
        key_gn2_bias = f"{n_block}/{n_unit}/gn2/bias"
        key_gn3_scale = f"{n_block}/{n_unit}/gn3/scale"
        key_gn3_bias = f"{n_block}/{n_unit}/gn3/bias"

        conv1_weight = np2th(weights[key_conv1], conv=True)
        conv2_weight = np2th(weights[key_conv2], conv=True)
        conv3_weight = np2th(weights[key_conv3], conv=True)
        gn1_weight = np2th(weights[key_gn1_scale])
        gn1_bias = np2th(weights[key_gn1_bias])
        gn2_weight = np2th(weights[key_gn2_scale])
        gn2_bias = np2th(weights[key_gn2_bias])
        gn3_weight = np2th(weights[key_gn3_scale])
        gn3_bias = np2th(weights[key_gn3_bias])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)
        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))
        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))
        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            key_proj_conv = f"{n_block}/{n_unit}/conv_proj/kernel"
            key_proj_gn_scale = f"{n_block}/{n_unit}/gn_proj/scale"
            key_proj_gn_bias = f"{n_block}/{n_unit}/gn_proj/bias"

            proj_conv_weight = np2th(weights[key_proj_conv], conv=True)
            proj_gn_weight = np2th(weights[key_proj_gn_scale])
            proj_gn_bias = np2th(weights[key_proj_gn_bias])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    def __init__(self, block_units, width_factor, in_channels=3):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(in_channels, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        x_root = self.root(x)
        features.append(x_root)
        
        x_pooled = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x_root)
        
        x_block1 = self.body[0](x_pooled)
        features.append(x_block1)
        
        x_block2 = self.body[1](x_block1)
        features.append(x_block2)
        
        x_final_vit_input = self.body[2](x_block2)
        
        return x_final_vit_input, features[::-1]


# --- Transformer Components ---
class Attention(nn.Module):
    def __init__(self, config, vis=False):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config['transformer']["num_heads"]
        self.attention_head_size = int(config['hidden_size'] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config['hidden_size'], self.all_head_size)
        self.key = Linear(config['hidden_size'], self.all_head_size)
        self.value = Linear(config['hidden_size'], self.all_head_size)

        self.out = Linear(config['hidden_size'], config['hidden_size'])
        self.attn_dropout = Dropout(config['transformer']["attention_dropout_rate"])
        self.proj_dropout = Dropout(config['transformer']["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config['hidden_size'], config['transformer']["mlp_dim"])
        self.fc2 = Linear(config['transformer']["mlp_dim"], config['hidden_size'])
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config['transformer']["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = True
        self.config = config
        img_size = _pair(img_size)

        grid_size = config['patches']["grid"]
        patch_size_conv = ( (img_size[0] // 16) // grid_size[0], (img_size[1] // 16) // grid_size[1] )
        
        n_patches = grid_size[0] * grid_size[1]
        
        self.hybrid_model = ResNetV2(block_units=config['resnet']['num_layers'],
                                     width_factor=config['resnet']['width_factor'],
                                     in_channels=in_channels)
        
        resnet_out_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=resnet_out_channels,
                                       out_channels=config['hidden_size'],
                                       kernel_size=patch_size_conv, 
                                       stride=patch_size_conv)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config['hidden_size']))
        self.dropout = Dropout(config['transformer']["dropout_rate"])

    def forward(self, x):
        x_resnet_out, features_skip = self.hybrid_model(x)
        x_patched = self.patch_embeddings(x_resnet_out)
        x_patched = x_patched.flatten(2)
        x_patched = x_patched.transpose(-1, -2)

        embeddings = x_patched + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features_skip

class Block(nn.Module):
    def __init__(self, config, vis=False):
        super(Block, self).__init__()
        self.hidden_size = config['hidden_size']
        self.attention_norm = LayerNorm(config['hidden_size'], eps=1e-6)
        self.ffn_norm = LayerNorm(config['hidden_size'], eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x_norm_attn = self.attention_norm(x)
        x_attn, weights = self.attn(x_norm_attn)
        x = x_attn + h

        h = x
        x_norm_ffn = self.ffn_norm(x)
        x_ffn = self.ffn(x_norm_ffn)
        x = x_ffn + h
        return x, weights

    def load_from(self, weights, n_block_idx_str):
        ROOT = f"Transformer/encoderblock_{n_block_idx_str}"
        
        key_q_kernel = f"{ROOT}/{ATTENTION_Q}/kernel"
        key_k_kernel = f"{ROOT}/{ATTENTION_K}/kernel"
        key_v_kernel = f"{ROOT}/{ATTENTION_V}/kernel"
        key_out_kernel = f"{ROOT}/{ATTENTION_OUT}/kernel"
        key_q_bias = f"{ROOT}/{ATTENTION_Q}/bias"
        key_k_bias = f"{ROOT}/{ATTENTION_K}/bias"
        key_v_bias = f"{ROOT}/{ATTENTION_V}/bias"
        key_out_bias = f"{ROOT}/{ATTENTION_OUT}/bias"

        key_fc0_kernel = f"{ROOT}/{FC_0}/kernel"
        key_fc1_kernel = f"{ROOT}/{FC_1}/kernel"
        key_fc0_bias = f"{ROOT}/{FC_0}/bias"
        key_fc1_bias = f"{ROOT}/{FC_1}/bias"

        key_attn_norm_scale = f"{ROOT}/{ATTENTION_NORM}/scale"
        key_attn_norm_bias = f"{ROOT}/{ATTENTION_NORM}/bias"
        key_mlp_norm_scale = f"{ROOT}/{MLP_NORM}/scale"
        key_mlp_norm_bias = f"{ROOT}/{MLP_NORM}/bias"

        with torch.no_grad():
            self.attn.query.weight.copy_(np2th(weights[key_q_kernel]).view(self.hidden_size, self.hidden_size).t())
            self.attn.key.weight.copy_(np2th(weights[key_k_kernel]).view(self.hidden_size, self.hidden_size).t())
            self.attn.value.weight.copy_(np2th(weights[key_v_kernel]).view(self.hidden_size, self.hidden_size).t())
            self.attn.out.weight.copy_(np2th(weights[key_out_kernel]).view(self.hidden_size, self.hidden_size).t())
            self.attn.query.bias.copy_(np2th(weights[key_q_bias]).view(-1))
            self.attn.key.bias.copy_(np2th(weights[key_k_bias]).view(-1))
            self.attn.value.bias.copy_(np2th(weights[key_v_bias]).view(-1))
            self.attn.out.bias.copy_(np2th(weights[key_out_bias]).view(-1))

            self.ffn.fc1.weight.copy_(np2th(weights[key_fc0_kernel]).t())
            self.ffn.fc2.weight.copy_(np2th(weights[key_fc1_kernel]).t())
            self.ffn.fc1.bias.copy_(np2th(weights[key_fc0_bias]).t())
            self.ffn.fc2.bias.copy_(np2th(weights[key_fc1_bias]).t())

            self.attention_norm.weight.copy_(np2th(weights[key_attn_norm_scale]))
            self.attention_norm.bias.copy_(np2th(weights[key_attn_norm_bias]))
            self.ffn_norm.weight.copy_(np2th(weights[key_mlp_norm_scale]))
            self.ffn_norm.bias.copy_(np2th(weights[key_mlp_norm_bias]))

class Encoder(nn.Module):
    def __init__(self, config, vis=False):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config['hidden_size'], eps=1e-6)
        for _ in range(config['transformer']["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Transformer(nn.Module):
    def __init__(self, config, img_size, in_channels, vis=False):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size, in_channels=in_channels)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features_skip = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features_skip

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                 skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1, activation=None):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        up = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        
        layers = [conv2d, up]
        if activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'softmax':
            layers.append(nn.Softmax(dim=1))
        super().__init__(*layers)

class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(config['hidden_size'], head_channels, kernel_size=3, padding=1, use_batchnorm=True)
        
        decoder_channels = config['decoder_channels']
        in_channels_cup = [head_channels] + list(decoder_channels[:-1])
        out_channels_cup = decoder_channels
        
        cfg_skip_channels = config['skip_channels']
        
        self.blocks = nn.ModuleList()
        for i in range(len(decoder_channels)):
            skip_ch_for_this_block = 0
            if i < config['n_skip'] and i < len(cfg_skip_channels):
                skip_ch_for_this_block = cfg_skip_channels[i]

            block = DecoderBlock(in_channels_cup[i], out_channels_cup[i], skip_channels=skip_ch_for_this_block)
            self.blocks.append(block)
            
    def forward(self, hidden_states, features_skip=None):
        B, n_patch, hidden = hidden_states.size()
        h_grid, w_grid = int(math.sqrt(n_patch)), int(math.sqrt(n_patch))

        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h_grid, w_grid)
        x = self.conv_more(x)

        for i, decoder_block in enumerate(self.blocks):
            skip_to_use = None
            if features_skip is not None and i < self.config['n_skip'] and i < len(features_skip):
                skip_to_use = features_skip[i]
            x = decoder_block(x, skip=skip_to_use)
        return x

class TransUNet(nn.Module):
    def __init__(self, config_name='R50-ViT-B_16', img_size=256, in_channels=3, num_classes=1,
                 load_resnet_weights=True, output_activation='sigmoid'): # MODIFIED PARAMETER
        super(TransUNet, self).__init__()
        
        if config_name not in CONFIGS:
            raise ValueError(f"Config {config_name} not found. Available: {list(CONFIGS.keys())}")
        
        self.vit_config = copy.deepcopy(CONFIGS[config_name])
        self.vit_config['n_classes'] = num_classes
        
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.load_resnet_weights = load_resnet_weights # STORE THE FLAG

        self.transformer = Transformer(self.vit_config, img_size=self.img_size, in_channels=self.in_channels)
        self.decoder = DecoderCup(self.vit_config)
        self.segmentation_head = SegmentationHead(
            in_channels=self.vit_config['decoder_channels'][-1],
            out_channels=self.vit_config['n_classes'],
            kernel_size=3,
            activation=output_activation
        )

        # MODIFIED: Always attempt to load NPZ if path exists.
        # The load_from_npz method will use self.load_resnet_weights internally.
        pretrained_path = self.vit_config['pretrained_path']
        if Path(pretrained_path).is_file():
            print(f"Attempting to load weights from {pretrained_path}")
            try:
                self.load_from_npz(weights_np=np.load(pretrained_path))
            except Exception as e:
                print(f"ERROR loading NPZ weights: {e}. Model weights may be partially loaded or random.")
        else:
            print(f"WARNING: Pretrained weights not found at {pretrained_path}. Model initialized randomly.")

    def forward(self, x):
        if x.size(1) != self.in_channels:
            if self.in_channels == 3 and x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)

        x_tf, attn_weights, features_resnet_skip = self.transformer(x)
        x_decoder = self.decoder(x_tf, features_skip=features_resnet_skip)
        logits = self.segmentation_head(x_decoder)
        return logits

    def load_from_npz(self, weights_np):
        with torch.no_grad():
            # --- Load ResNet Weights (Conditional) ---
            if self.load_resnet_weights:
                if self.transformer.embeddings.hybrid:
                    print("Attempting to load ResNet weights (load_resnet_weights=True, model is hybrid).")
                    hybrid_model = self.transformer.embeddings.hybrid_model
                    
                    # ResNet Root
                    if self.in_channels == 3:
                        if "conv_root/kernel" in weights_np:
                            hybrid_model.root.conv.weight.copy_(np2th(weights_np["conv_root/kernel"], conv=True))
                            hybrid_model.root.gn.weight.copy_(np2th(weights_np["gn_root/scale"]).view(-1))
                            hybrid_model.root.gn.bias.copy_(np2th(weights_np["gn_root/bias"]).view(-1))
                            print("ResNet root weights loaded.")
                        else:
                            print("WARNING: Skipping ResNet root: 'conv_root/kernel' not in weights_np (possibly dummy file).")
                    else:
                        print(f"WARNING: Skipping ResNet root loading: model in_channels ({self.in_channels}) != 3. Pretrained ResNet root expects 3 input channels.")

                    # ResNet Body
                    resnet_body_loaded_any = False
                    for bname, block_module in hybrid_model.body.named_children():
                        for uname, unit_module in block_module.named_children():
                            first_key_check = f"{bname}/{uname}/conv1/kernel"
                            if first_key_check in weights_np:
                                unit_module.load_from(weights_np, n_block=bname, n_unit=uname)
                                resnet_body_loaded_any = True
                    if resnet_body_loaded_any:
                        print("ResNet body weights loaded (or attempted for relevant units).")
                    else:
                        print("WARNING: No ResNet body unit weights were loaded (keys might be missing for all units).")

                else: # Not hybrid
                    print("NOTE: Skipping ResNet weights: Model is not hybrid, though load_resnet_weights=True.")
            else: # load_resnet_weights is False
                print("NOTE: Skipping ResNet weights as load_resnet_weights=False.")

            # --- Load Transformer-specific Weights (Always Attempted) ---
            print("Attempting to load Transformer-specific weights.")
            if "embedding/kernel" in weights_np:
                self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights_np["embedding/kernel"], conv=True))
                self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights_np["embedding/bias"]))
                print("Transformer patch_embeddings weights loaded.")
            else:
                print("WARNING: 'embedding/kernel' not found for Transformer patch_embeddings.")
            
            if "Transformer/encoder_norm/scale" in weights_np:
                self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights_np["Transformer/encoder_norm/scale"]))
                self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights_np["Transformer/encoder_norm/bias"]))
                print("Transformer encoder_norm weights loaded.")
            else:
                print("WARNING: 'Transformer/encoder_norm/scale' not found.")


            if "Transformer/posembed_input/pos_embedding" in weights_np:
                posemb_orig = np2th(weights_np["Transformer/posembed_input/pos_embedding"])
                posemb_new = self.transformer.embeddings.position_embeddings
                
                n_tok_orig = posemb_orig.shape[1]
                n_tok_new = posemb_new.shape[1]
                
                posemb_to_resize = posemb_orig
                if n_tok_orig == int(math.sqrt(n_tok_orig -1))**2 + 1 :
                    print(f"Original position embedding has {n_tok_orig} tokens. Assuming one is a CLS token. Stripping it.")
                    posemb_to_resize = posemb_orig[:, 1:]
                elif n_tok_orig == int(math.sqrt(n_tok_orig))**2 :
                    print(f"Original position embedding has {n_tok_orig} tokens, forming a square grid.")
                else:
                    logger.warning(f"Original position embedding token count ({n_tok_orig}) is unusual. Proceeding without stripping CLS explicitly here.")

                if posemb_to_resize.shape[1] == n_tok_new:
                    print("Position embedding token count matches target. Copying directly.")
                    self.transformer.embeddings.position_embeddings.copy_(posemb_to_resize)
                else: 
                    logger.info(f"Resizing position embeddings from {posemb_to_resize.shape} to target token count {n_tok_new}")
                    posemb_grid_squeezed = posemb_to_resize.squeeze(0)
                    
                    if int(math.sqrt(posemb_grid_squeezed.shape[0]))**2 != posemb_grid_squeezed.shape[0]:
                        logger.error(f"Cannot form a square grid from source position embedding of shape {posemb_grid_squeezed.shape} for resizing. Skipping pos_embed loading.")
                        print("ERROR: Position embedding resizing failed, cannot form square grid from source.")
                    else:
                        gs_old = int(math.sqrt(posemb_grid_squeezed.shape[0]))
                        gs_new = int(math.sqrt(n_tok_new))

                        if gs_old == 0 or gs_new == 0:
                            logger.error("Grid size old or new is zero. Skipping pos_embed loading.")
                            print("ERROR: Position embedding resizing failed, grid size zero.")
                        else:
                            print(f"Resizing position embedding grid from {gs_old}x{gs_old} to {gs_new}x{gs_new}.")
                            posemb_reshaped_old = posemb_grid_squeezed.reshape(gs_old, gs_old, -1)
                            zoom_factors = (gs_new / gs_old, gs_new / gs_old, 1)
                            posemb_grid_zoomed = ndimage.zoom(posemb_reshaped_old, zoom_factors, order=1)
                            posemb_final_reshaped = posemb_grid_zoomed.reshape(1, gs_new * gs_new, -1)
                            self.transformer.embeddings.position_embeddings.copy_(torch.from_numpy(posemb_final_reshaped))
                            print("Transformer position_embeddings resized and loaded.")
            else:
                print("WARNING: 'Transformer/posembed_input/pos_embedding' not found. Position embeddings initialized randomly.")

            transformer_blocks_loaded_any = False
            for idx, block_module in enumerate(self.transformer.encoder.layer):
                first_key_check = f"Transformer/encoderblock_{idx}/{ATTENTION_Q}/kernel"
                if first_key_check in weights_np:
                    block_module.load_from(weights_np, n_block_idx_str=str(idx))
                    transformer_blocks_loaded_any = True
            if transformer_blocks_loaded_any:
                print("Transformer encoder blocks loaded (or attempted for relevant blocks).")
            else:
                print("WARNING: No Transformer encoder block weights were loaded (keys might be missing).")

        print("Finished processing pre-trained weights for TransUNet.")
