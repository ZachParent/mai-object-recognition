from enum import Enum
from pathlib import Path
from typing import Optional

import pydantic


class ModelName(str, Enum):
    UNET2D = "unet2d"


class UNet2DConfig(pydantic.BaseModel):
    input_size: tuple[int, int, int] = (256, 256, 3)
    filter_num: list[int] = [64, 128, 256, 512]
    n_labels: int = 1  # For depth estimation, we only need one channel


class RunConfig(pydantic.BaseModel):
    id: int
    model_name: ModelName
    learning_rate: float
    batch_size: int = 16
    epochs: int = 2
    augmentation: bool = False
    save_path: Optional[Path] = None
    unet2d_config: Optional[UNet2DConfig] = None
    save_img_ids: list[int] = []
    seed: Optional[int] = None


class RunSet(pydantic.BaseModel):
    title: str
    configs: list[RunConfig]


BASE_RUN_SET = RunSet(
    title="Base",
    configs=[
        RunConfig(
            id=1,
            model_name="unet2d",
            learning_rate=0.0001,
            batch_size=16,
            augmentation=False,
            save_img_ids=[1, 5],
        ),
    ],
)
