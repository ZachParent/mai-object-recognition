from typing import Literal

import pydantic

ModelName = Literal["unet2d"]


class RunConfig(pydantic.BaseModel):
    id: int
    model_name: ModelName
    learning_rate: float
    batch_size: int = 16
    augmentation: bool = False
    save_img_ids: list[int] = []


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
