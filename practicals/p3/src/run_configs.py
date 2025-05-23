import itertools
from enum import Enum
from typing import Literal, Optional

import pandas as pd
import pydantic

from .config import RESULTS_DIR


class ModelName(str, Enum):
    UNET2D = "unet2d"
    ATTENTION_UNET = "attention_unet"
    RESUNET_A = "resunet_a"
    TRANSUNET = "transunet"


class UNet2DConfig(pydantic.BaseModel):
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
    input_size: tuple[int, int, int] = (256, 256, 3)
    batch_size: int = 64
    epochs: int = 12
    augmentation: bool = False
    perceptual_loss: Literal["L1", "L2"] = "L2"
    perceptual_loss_weight: Optional[float] = None
    save_model: bool = True
    unet2d_config: Optional[UNet2DConfig] = None
    include_pose: bool = False
    seed: Optional[int] = None
    transunet_use_pretrained: Optional[bool] = None


class RunSet(pydantic.BaseModel):
    title: str
    configs: list[RunConfig]


SEEDS = [0, 1, 2]

id_start = 0

HYPERPARAM_RUN_SET = RunSet(
    title="Hyperparameter Tuning",
    configs=[
        RunConfig(
            id=id_start + i,
            name="Base",
            epochs=12,
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,
            unet2d_config=UNet2DConfig(),
            seed=seed,
        )
        for i, seed in enumerate(SEEDS)
    ]
    + [
        RunConfig(
            id=id_start + 1 * len(SEEDS) + i,
            name="Reduced Depth",
            epochs=12,
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,
            unet2d_config=UNet2DConfig(
                filter_num=[64, 128, 256, 1024],
            ),
            seed=seed,
        )
        for i, seed in enumerate(SEEDS)
    ]
    + [
        RunConfig(
            id=id_start + 2 * len(SEEDS) + i,
            name="No Batch Norm",
            epochs=12,
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,
            unet2d_config=UNet2DConfig(batch_norm=False),
            seed=seed,
        )
        for i, seed in enumerate(SEEDS)
    ]
    + [
        RunConfig(
            id=id_start + 3 * len(SEEDS) + i,
            name="Higher Learning Rate",
            epochs=12,
            model_name=ModelName.UNET2D,
            learning_rate=0.0003,
            unet2d_config=UNet2DConfig(),
            seed=seed,
        )
        for i, seed in enumerate(SEEDS)
    ]
    + [
        RunConfig(
            id=id_start + 4 * len(SEEDS) + i,
            name="Lower Learning Rate",
            epochs=12,
            model_name=ModelName.UNET2D,
            learning_rate=0.00003,
            unet2d_config=UNet2DConfig(),
            seed=seed,
        )
        for i, seed in enumerate(SEEDS)
    ]
    + [
        RunConfig(
            id=id_start + 5 * len(SEEDS) + i,
            name="With augmentation",
            epochs=12,
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,
            augmentation=True,
            unet2d_config=UNet2DConfig(),
            seed=seed,
        )
        for i, seed in enumerate(SEEDS)
    ],
)

id_start = 20

VIT_RUN_SET = RunSet(
    title="Vision Transformer",
    configs=[
        RunConfig(
            id=id_start + i,
            name="Attention UNet",
            model_name=ModelName.ATTENTION_UNET,
            learning_rate=0.0001,
            seed=seed,
        )
        for i, seed in enumerate(SEEDS)
    ]
    + [
        RunConfig(
            id=id_start + 1 * len(SEEDS) + i,
            name="ResUNet A",
            model_name=ModelName.RESUNET_A,
            learning_rate=0.0001,
            seed=seed,
        )
        for i, seed in enumerate(SEEDS)
    ]
    + [
        RunConfig(
            id=id_start + 2 * len(SEEDS) + i,
            name="TransUNet (no pretraining)",
            model_name=ModelName.TRANSUNET,
            learning_rate=0.0001,
            transunet_use_pretrained=False,
            seed=seed,
        )
        for i, seed in enumerate(SEEDS)
    ]
    + [
        RunConfig(
            id=id_start + 3 * len(SEEDS) + i,
            name="TransUNet (pretrained)",
            model_name=ModelName.TRANSUNET,
            learning_rate=0.0001,
            transunet_use_pretrained=True,
            seed=seed,
        )
        for i, seed in enumerate(SEEDS)
    ],
)

id_start = 40

UNET2D_PERCEPTUAL_LOSS_RUN_SET = RunSet(
    title="Perceptual Loss UNet2D",
    configs=[
        RunConfig(
            id=id_start + i,
            name=f"UNet2D {weight} {l1_l2} perceptual {1 - weight} MSE",
            epochs=12,
            # TODO: use the best network config
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,
            unet2d_config=UNet2DConfig(),
            perceptual_loss_weight=weight,
            perceptual_loss=l1_l2,  # type: ignore
            seed=seed,
        )
        for i, (weight, l1_l2, seed) in enumerate(
            itertools.product([0.25, 0.5, 0.75], ["L1", "L2"], SEEDS)
        )
    ],
)

id_start = 60

TRANSUNET_PERCEPTUAL_LOSS_RUN_SET = RunSet(
    title="Perceptual Loss TransUNet",
    configs=[
        RunConfig(
            id=id_start + i,
            name=f"TransUNet {0.5} {l1_l2} perceptual {0.5} MSE",
            epochs=12,
            # TODO: use the best network config
            model_name=ModelName.TRANSUNET,
            transunet_use_pretrained=True,
            learning_rate=0.0001,
            perceptual_loss_weight=0.5,
            perceptual_loss=l1_l2,  # type: ignore
            seed=seed,
        )
        for i, (l1_l2, seed) in enumerate(itertools.product(["L1", "L2"], SEEDS))
    ],
)

id_start = 80

SMPL_RUN_SET = RunSet(
    title="SMPL",
    configs=[
        RunConfig(
            id=id_start + i,
            name=f"TransUNet {'0.5 L2 perceptual 0.5 MSE ' if weight is not None else ''}with SMPL",
            model_name=ModelName.TRANSUNET,
            learning_rate=0.0001,
            input_size=(256, 256, 6),
            transunet_use_pretrained=True,
            perceptual_loss="L2",
            perceptual_loss_weight=weight,
            include_pose=True,
            seed=seed,
        )
        for i, (weight, seed) in enumerate(itertools.product([None, 0.5], SEEDS))
    ],
)

ALL_RUNS_SETS = [
    HYPERPARAM_RUN_SET,
    VIT_RUN_SET,
    UNET2D_PERCEPTUAL_LOSS_RUN_SET,
    TRANSUNET_PERCEPTUAL_LOSS_RUN_SET,
    SMPL_RUN_SET,
]


if __name__ == "__main__":
    run_dfs = []
    for run_set in ALL_RUNS_SETS:
        new_df = pd.DataFrame([run.model_dump() for run in run_set.configs])
        if new_df.empty:
            continue
        new_df["run_set"] = run_set.title
        new_df["id"] = new_df["id"].astype(int)
        new_df["json"] = [run.model_dump_json() for run in run_set.configs]
        run_dfs.append(new_df)
    df = pd.concat(run_dfs, ignore_index=True)
    df.to_csv(RESULTS_DIR / "run_configs.csv", index=False)
