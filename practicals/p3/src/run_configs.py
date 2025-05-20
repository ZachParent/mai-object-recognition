import itertools
from enum import Enum
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
    epochs: int = 12
    augmentation: bool = False
    perceptual_loss: Literal["L1", "L2"] = "L2"
    perceptual_loss_weight: Optional[float] = None
    save_model: bool = True
    unet2d_config: Optional[UNet2DConfig] = None
    seed: Optional[int] = None


class RunSet(pydantic.BaseModel):
    title: str
    configs: list[RunConfig]


id_start = 0


# TODO: use 3 seeds each run throughout
HYPERPARAM_RUN_SET = RunSet(
    title="Hyperparameter Tuning",
    configs=[
        RunConfig(
            id=id_start,
            name="Base",
            epochs=12,
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,
            unet2d_config=UNet2DConfig(),
        ),
        RunConfig(
            id=id_start + 1,
            name="Reduced Depth",
            epochs=12,
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,
            unet2d_config=UNet2DConfig(
                filter_num=[64, 128, 256, 1024],
            ),
        ),
        RunConfig(
            id=id_start + 2,
            name="No Batch Norm",
            epochs=12,
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,
            unet2d_config=UNet2DConfig(batch_norm=False),
        ),
        RunConfig(
            id=id_start + 3,
            name="Higher Learning Rate",
            epochs=12,
            model_name=ModelName.UNET2D,
            learning_rate=0.0003,
            unet2d_config=UNet2DConfig(),
        ),
        RunConfig(
            id=id_start + 4,
            name="Lower Learning Rate",
            epochs=12,
            model_name=ModelName.UNET2D,
            learning_rate=0.00003,
            unet2d_config=UNet2DConfig(),
        ),
        RunConfig(
            id=id_start + 5,
            name="With augmentation",
            epochs=12,
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,
            augmentation=True,
            unet2d_config=UNet2DConfig(),
        ),
    ],
)

VIT_RUN_SET = None  # TODO: add vit runs

# 1 per architecture

PERCEPTUAL_LOSS_RUN_SET = RunSet(
    title="Perceptual Loss",
    configs=[
        RunConfig(
            id=id_start,
            name=f"{weight} {l1_l2} perceptual {1 - weight} MSE - seed: {seed}",
            epochs=12,
            # TODO: use the best network config
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,
            unet2d_config=UNet2DConfig(),
            perceptual_loss_weight=weight,
            perceptual_loss=l1_l2,  # type: ignore
            seed=seed,
        )
        for weight, l1_l2, seed in itertools.product(
            [0.25, 0.5, 0.75], ["L1", "L2"], [0, 1, 2]
        )
    ],
)

SMPL_RUN_SET = RunSet(
    title="SMPL",
    configs=[
        RunConfig(
            id=id_start,
            name="SMPL",
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,
            unet2d_config=UNet2DConfig(input_size=(256, 256, 6)),
            seed=seed,
            # TODO: use a flag to add pose information
        )
        for seed in [0, 1, 2]
    ],
)

id_start += len(HYPERPARAM_RUN_SET.configs)
