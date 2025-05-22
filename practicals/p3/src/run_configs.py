# src/run_configs.py
import itertools
from enum import Enum
from typing import Literal, Optional

import pandas as pd
import pydantic

from config import RESULTS_DIR # Assuming this is in src/config.py relative to run_configs.py


class ModelName(str, Enum):
    UNET2D = "unet2d"
    TRANSUNET = "transunet" # Added


class UNet2DConfig(pydantic.BaseModel):
    filter_num: list[int] = [64, 128, 256, 512, 1024]
    n_labels: int = 1
    stack_num_down: int = 2
    stack_num_up: int = 1
    activation: str = "GELU"
    output_activation: str = "Sigmoid"
    batch_norm: bool = True
    pool: bool = True
    unpool: bool = False


class TransUNetModelConfig(pydantic.BaseModel): # Renamed to avoid conflict
    name: str = "R50-ViT-B_16" # e.g., 'R50-ViT-B_16'
    img_size: int = 256
    # in_channels will be taken from RunConfig.input_size[2]
    num_classes: int = 1 # For depth
    load_pretrained: bool = True
    output_activation: str = "sigmoid" # Matches U-Net default for depth


class RunConfig(pydantic.BaseModel):
    id: int
    name: str
    model_name: ModelName
    learning_rate: float
    input_size: tuple[int, int, int] = (256, 256, 3) # (H, W, C)
    batch_size: int = 64
    epochs: int = 12
    augmentation: bool = False
    perceptual_loss: Literal["L1", "L2"] = "L2"
    perceptual_loss_weight: Optional[float] = None
    save_model: bool = True
    unet2d_config: Optional[UNet2DConfig] = None
    transunet_config: Optional[TransUNetModelConfig] = None # Added
    include_pose: bool = False # This will affect input_size[2] if True
    seed: Optional[int] = None


class RunSet(pydantic.BaseModel):
    title: str
    configs: list[RunConfig]


SEEDS = [0, 1, 2]
id_start = 0 # Define id_start if not already defined

# ... (Your existing HYPERPARAM_RUN_SET, etc.)

# Example of adding a TransUNet run set
# Ensure id_start is correctly managed if you have other run sets
# For example, if HYPERPARAM_RUN_SET was defined and used id_start:
# id_start += len(HYPERPARAM_RUN_SET.configs)

TRANSUNET_RUN_SET = RunSet(
    title="TransUNet Basic",
    configs=[
        RunConfig(
            id=id_start + i, # Make sure id_start is correct
            name=f"TransUNet_R50ViTB16_seed{seed}",
            model_name=ModelName.TRANSUNET,
            learning_rate=0.0001, # Adjust as needed
            input_size=(256, 256, 3), # For RGB input
            batch_size=4, # TransUNet is larger, might need smaller batch size
            epochs=20,    # May need more epochs
            transunet_config=TransUNetModelConfig(name="R50-ViT-B_16", load_pretrained=True),
            seed=seed,
            include_pose=False # Example for 3-channel input
        )
        for i, seed in enumerate(SEEDS)
    ]
)
id_start += len(TRANSUNET_RUN_SET.configs)

# Add SMPL (6-channel input) example for TransUNet
TRANSUNET_SMPL_RUN_SET = RunSet(
    title="TransUNet SMPL",
    configs=[
        RunConfig(
            id=id_start + i,
            name=f"TransUNet_R50ViTB16_SMPL_seed{seed}",
            model_name=ModelName.TRANSUNET,
            learning_rate=0.0001,
            input_size=(256, 256, 6), # For RGB + Pose input
            batch_size=4,
            epochs=20,
            # For 6-channel input, pre-trained ResNet weights (for the root conv) won't directly apply.
            # The Transformer part can still be pre-trained.
            # load_pretrained=False if the ResNet part cannot take pre-trained weights for 6ch.
            # Or, you'd need a custom loading that only loads compatible parts.
            # The current TransUNet load_from_npz will try to load ResNet weights.
            # Setting load_pretrained=False for the whole TransUNet config is safer if first ResNet layer is changed.
            transunet_config=TransUNetModelConfig(name="R50-ViT-B_16", load_pretrained=False),
            seed=seed,
            include_pose=True
        )
        for i, seed in enumerate(SEEDS)
    ]
)
id_start += len(TRANSUNET_SMPL_RUN_SET.configs)


ALL_RUNS_SETS = [
    # HYPERPARAM_RUN_SET,
    # VIT_RUN_SET, # Your existing sets
    # PERCEPTUAL_LOSS_RUN_SET,
    # SMPL_RUN_SET,
    TRANSUNET_RUN_SET, # Add new set
    # TRANSUNET_SMPL_RUN_SET
]


if __name__ == "__main__":
    run_dfs = []
    for run_set in ALL_RUNS_SETS:
        # Ensure HYPERPARAM_RUN_SET is defined or handle it
        if not hasattr(run_set, 'configs') or not run_set.configs:
            print(f"Skipping empty or undefined run_set: {run_set.title if hasattr(run_set, 'title') else 'Unknown'}")
            continue
        new_df = pd.DataFrame([run.model_dump() for run in run_set.configs])
        if new_df.empty:
            continue
        new_df["run_set"] = run_set.title
        new_df["id"] = new_df["id"].astype(int)

        if "unet2d_config" in new_df.columns:
            new_df["unet2d_filter_num"] = new_df["unet2d_config"].apply(
                lambda x: x["filter_num"] if x else None # Handle None case
            )
            new_df = new_df.drop(columns=["unet2d_config"])
        if "transunet_config" in new_df.columns: # Added for transunet
            new_df["transunet_model_name_cfg"] = new_df["transunet_config"].apply(
                lambda x: x["name"] if x else None
            )
            new_df = new_df.drop(columns=["transunet_config"])

        new_df["json"] = [run.model_dump_json() for run in run_set.configs]
        run_dfs.append(new_df)

    if run_dfs:
        df = pd.concat(run_dfs, ignore_index=True)
        df.to_csv(RESULTS_DIR / "run_configs.csv", index=False)
        print("Saved run_configs.csv")
    else:
        print("No run configurations to save.")