from enum import Enum
from pathlib import Path
from typing import Optional

import pydantic


class ModelName(str, Enum):
    UNET2D = "unet2d"


class UNet2DConfig(pydantic.BaseModel):
    input_size: tuple[int, int, int] = (256, 256, 3)
    filter_num: list[int] = [64, 128, 256, 512, 1024]
    n_labels: int = 1  # For depth estimation, we only need one channel
    stack_num_down: int = 2
    stack_num_up: int = 1
    activation: str = "GELU"
    output_activation: str = "Sigmoid"
    batch_norm: bool = True
    pool: bool = True
    unpool: bool = False


class RunConfig(pydantic.BaseModel):
    id: int
    model_name: ModelName
    learning_rate: float
    batch_size: int = 16
    epochs: int = 2
    augmentation: bool = False
    save_path: Optional[Path] = None
    unet2d_config: Optional[UNet2DConfig] = None
    save_video_ids: list[int] = []
    seed: Optional[int] = None


class RunSet(pydantic.BaseModel):
    title: str
    configs: list[RunConfig]


BASE_RUN_SET = RunSet(
    title="Base",
    configs=[
        RunConfig(
            id=1,
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,
            batch_size=16,
            augmentation=False,
            unet2d_config=UNet2DConfig(),  # Default config
            save_video_ids=[0],
        ),
    ],
)

HYPERPARAM_RUN_SET = RunSet(
    title="Hyperparameter Tuning",
    configs=[
        RunConfig(
            id=2,
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,
            batch_size=16,
            augmentation=False,
            unet2d_config=UNet2DConfig(
                filter_num=[64, 256, 1024],  # Reduced depth, increased width
            ),
            save_video_ids=[0],
        ),
        RunConfig(
            id=3,
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,
            batch_size=16,
            augmentation=False,
            unet2d_config=UNet2DConfig(batch_norm=False),  # No batch normalization
            save_video_ids=[0],
        ),
    ],
)
