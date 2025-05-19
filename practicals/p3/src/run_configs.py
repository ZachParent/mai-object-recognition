from enum import Enum
from pathlib import Path
from typing import Literal, Optional

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
    name: str
    model_name: ModelName
    learning_rate: float
    batch_size: int = 64
    epochs: int = 6
    augmentation: bool = False
    perceptual_loss: Literal["L1", "L2"] = "L2"
    perceptual_loss_weight: Optional[float] = None
    save_path: Optional[Path] = None
    unet2d_config: Optional[UNet2DConfig] = None
    save_video_ids: list[int] = []
    seed: Optional[int] = None


class RunSet(pydantic.BaseModel):
    title: str
    configs: list[RunConfig]


id_start = 0

HYPERPARAM_RUN_SET = RunSet(
    title="Hyperparameter Tuning",
    configs=[
        RunConfig(
            id=id_start,
            name="Base",
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,
            unet2d_config=UNet2DConfig(),
            save_video_ids=[0],
            seed=seed,
        )
        for seed in range(3)
    ]
    + [
        RunConfig(
            id=id_start + 1,
            name="Reduced Depth",
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,
            unet2d_config=UNet2DConfig(
                filter_num=[64, 256, 1024],
            ),
            save_video_ids=[0],
            seed=seed,
        )
        for seed in range(3)
    ]
    + [
        RunConfig(
            id=id_start + 2,
            name="No Batch Norm",
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,
            unet2d_config=UNet2DConfig(batch_norm=False),
            save_video_ids=[0],
            seed=seed,
        )
        for seed in range(3)
    ]
    + [
        RunConfig(
            id=id_start + 3,
            name="Higher Learning Rate",
            model_name=ModelName.UNET2D,
            learning_rate=0.0003,
            unet2d_config=UNet2DConfig(),
            save_video_ids=[0],
            seed=seed,
        )
        for seed in range(3)
    ]
    + [
        RunConfig(
            id=id_start + 4,
            name="Lower Learning Rate",
            model_name=ModelName.UNET2D,
            learning_rate=0.00003,
            unet2d_config=UNet2DConfig(),
            save_video_ids=[0],
            seed=seed,
        )
        for seed in range(3)
    ]
    + [
        RunConfig(
            id=id_start + 5,
            name="With augmentation",
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,
            augmentation=True,
            unet2d_config=UNet2DConfig(),
            save_video_ids=[0],
            seed=seed,
        )
        for seed in range(3)
    ],
)


id_start += len(HYPERPARAM_RUN_SET.configs)
