# src/models/__init__.py

import torch.nn as nn

from ..run_configs import ModelName, RunConfig
from .attentionUnet import AttentionUNet
from .resUnet_a import ResUnetA
from .transUnet import TransUNet
from .unet2d import UNet2D


def get_model(config: RunConfig) -> nn.Module:
    if config.model_name == ModelName.UNET2D:
        if config.unet2d_config is None:
            raise ValueError("UNet2DConfig is required for UNET2D model.")
        return UNet2D(
            input_size=config.input_size,  # (H, W, C) tuple
            filter_num=config.unet2d_config.filter_num,
            n_labels=config.unet2d_config.n_labels,
            stack_num_down=config.unet2d_config.stack_num_down,
            stack_num_up=config.unet2d_config.stack_num_up,
            activation=config.unet2d_config.activation,
            output_activation=config.unet2d_config.output_activation,
            batch_norm=config.unet2d_config.batch_norm,
            pool=config.unet2d_config.pool,
            unpool=config.unet2d_config.unpool,
        )
    elif config.model_name == ModelName.ATTENTION_UNET:
        return AttentionUNet(
            input_size=config.input_size,
        )
    elif config.model_name == ModelName.RESUNET_A:
        return ResUnetA(
            input_size=config.input_size,
        )
    elif config.model_name == ModelName.TRANSUNET:
        img_size, _, in_channels = config.input_size

        assert (
            config.transunet_use_pretrained is not None
        ), "transunet_use_pretrained must be provided"
        return TransUNet(
            img_size=img_size,
            in_channels=in_channels,
            load_resnet_weights=config.transunet_use_pretrained,
        )
    else:
        raise ValueError(f"Model {config.model_name} not found")
