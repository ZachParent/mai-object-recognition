# src/models/__init__.py

import torch.nn as nn

# Ensure correct relative import for run_configs
# If models is a subfolder of src, and run_configs is in src:
from ..run_configs import ModelName, RunConfig, UNet2DConfig, TransUNetModelConfig
# If models and run_configs are siblings under a parent directory (e.g. project_root/src/models, project_root/src/run_configs.py)
# then the import might be:
# from ..run_configs import ModelName, RunConfig, UNet2DConfig, TransUNetModelConfig
# Or if run_configs.py is at the same level as the folder containing 'models'
# from ..run_configs import ... (adjust .. based on your exact structure)


from .unet2d import UNet2D
from .transunet import TransUNet # Import the TransUNet class


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
        if config.transunet_config is None:
            raise ValueError("TransUNetModelConfig is required for TRANSUNET model.")
        
        # The TransUNet model expects in_channels from config.input_size
        # config.input_size is (H, W, C)
        in_channels = config.input_size[2] 
        
        return TransUNet(
            config_name=config.transunet_config.name,
            img_size=config.transunet_config.img_size, # Should be consistent with config.input_size[0] and [1]
            in_channels=in_channels,
            num_classes=config.transunet_config.num_classes,
            load_pretrained_weights=config.transunet_config.load_pretrained,
            output_activation=config.transunet_config.output_activation
        )
    else:
        raise ValueError(f"Model {config.model_name} not found")