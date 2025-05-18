from models.unet2d import UNet2D
from run_configs import ModelName, RunConfig


def get_model(config: RunConfig) -> UNet2D:
    if config.model_name == ModelName.UNET2D:
        if config.unet2d_config is None:
            raise ValueError("UNet2D config is required")
        return UNet2D(
            input_size=config.unet2d_config.input_size,
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
    else:
        raise ValueError(f"Model {config.model_name} not found")
