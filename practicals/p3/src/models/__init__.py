# src/models/__init__.py

import torch.nn as nn

from ..run_configs import ModelName, RunConfig, UNet2DConfig
from .unet2d import UNet2D
from .transUnet import TransUNet # Import the TransUNet class


def get_model(config: RunConfig) -> nn.Module:
    if config.model_name == ModelName.UNET2D:
        if config.unet2d_config is None:
            raise ValueError("UNet2DConfig is required for UNET2D model.")
        return UNet2D(
            input_size=config.input_size, # (H, W, C) tuple
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
    elif config.model_name == ModelName.TRANSUNET: # Added condition for TransUNet
        
        # The TransUNet model expects in_channels from config.input_size
        # config.input_size is (H, W, C)
        in_channels = config.input_size[2] 
        
        return TransUNet(
            load_resnet_weights=config.transunet_use_pretrained,
        )
    else:
        raise ValueError(f"Model {config.model_name} not found")