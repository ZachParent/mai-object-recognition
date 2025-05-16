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
        )
    else:
        raise ValueError(f"Model {config.model_name} not found")
