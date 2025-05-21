import re
from pathlib import Path
from typing import List, Optional, Tuple

import einops
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from ..config import PREPROCESSED_DATA_DIR, RESULTS_DIR

with open(RESULTS_DIR / "pixel_distribution.csv", "r") as f:
    df = pd.read_csv(f)
    MEAN = df["mean"].tolist()
    STD = df["std_dev"].tolist()

FLOAT_TRANSFORM = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
    ]
)

NORMALIZE_TRANSFORM = v2.Compose(
    [
        v2.Normalize(mean=MEAN, std=STD),
    ]
)

AUGMENT_TRANSFORM = v2.Compose(
    [
        v2.RandomRotation([-5, 5]),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        v2.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ]
)


def get_data_paths(
    start_idx: int,
    end_idx: Optional[int] = None,
    include_pose: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    # Get all parent folders and sort them
    parent_folders = sorted(PREPROCESSED_DATA_DIR.glob("0*"))
    if end_idx is None:
        end_idx = len(parent_folders)

    selected_folders = parent_folders[start_idx:end_idx]

    # Get image and depth paths only from selected folders
    image_paths = []
    depth_paths = []
    pose_paths = []
    for folder in selected_folders:
        image_paths.extend(sorted(folder.glob("rgb/*.png")))
        depth_paths.extend(sorted(folder.glob("depth/*.npy")))
        if include_pose:
            pose_paths.extend(sorted(folder.glob("pose/*.png")))

    assert len(image_paths) == len(depth_paths)
    if include_pose:
        assert len(image_paths) == len(pose_paths)
    return image_paths, depth_paths, pose_paths


def normalize_depth(depth: tv_tensors.Mask) -> tv_tensors.Mask:
    # the background value is the max value in the depth map
    bg_value = depth.max()
    bg_mask = depth == bg_value
    zero_mask = depth == 0
    # the max foreground value is the max of the non-background values
    max_fg_value = depth[~bg_mask].max()
    new_bg_value = max_fg_value * 1.1
    # set the background and zero values to the new background value
    # zero values only occur as a result of rotation, so we can just set them to the new background value
    depth[bg_mask | zero_mask] = new_bg_value
    # normalize the depth map to the range [0, 1]
    depth[...] = (depth - depth.min()) / (new_bg_value - depth.min())
    return tv_tensors.Mask(depth.to(torch.float32))


class Cloth3dDataset(Dataset):
    def __init__(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        enable_normalization: bool = True,
        enable_augmentation: bool = False,
        include_pose: bool = False,
    ):
        self.enable_normalization = enable_normalization
        self.enable_augmentation = enable_augmentation
        self.include_pose = include_pose

        # Get parent folders to calculate num_videos and store video_ids
        parent_folders = sorted(PREPROCESSED_DATA_DIR.glob("0*"))
        if end_idx is None:
            end_idx = len(parent_folders)
        self.num_videos = end_idx - start_idx
        self.video_ids = [
            int(folder.name) for folder in parent_folders[start_idx:end_idx]
        ]

        self.image_paths, self.depth_paths, self.pose_paths = get_data_paths(
            start_idx, end_idx, include_pose=include_pose
        )

    def _get_data_paths(self, idx: int) -> Tuple[str, str, str] | Tuple[str, str]:
        if self.include_pose:
            return (
                self.image_paths[idx],
                self.depth_paths[idx],
                self.pose_paths[idx],
            )
        else:
            return self.image_paths[idx], self.depth_paths[idx]

    def get_video_id(self, idx: int) -> int:
        """Get the video ID for a given dataset index."""
        path = Path(self.image_paths[idx])
        return int(path.parent.parent.name)

    def get_frame_id(self, idx: int) -> int:
        """Get the frame ID for a given dataset index."""
        path = Path(self.image_paths[idx])
        match = re.search(r"frame(\d+)", path.stem)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Could not extract frame id from {path.stem}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[tv_tensors.Image, tv_tensors.Mask]:
        input_data = torch.from_numpy(np.array(Image.open(self.image_paths[idx])))
        target_data = torch.from_numpy(np.load(self.depth_paths[idx]))

        # move channel to first dimension, and drop alpha channel
        input_data = einops.rearrange(input_data, "h w c -> c h w")[:3]
        target_data = einops.rearrange(target_data, "h w -> () h w")

        # Wrap raw tensors into tv_tensors.Image and tv_tensors.Mask
        # This allows v2 transforms to correctly identify and process them.
        input_tensor = tv_tensors.Image(input_data)
        target_tensor = tv_tensors.Mask(target_data)
        if self.include_pose:
            pose_data = torch.from_numpy(np.array(Image.open(self.pose_paths[idx])))
            pose_data = einops.rearrange(pose_data, "h w c -> c h w")[:3]
            pose_tensor = tv_tensors.Image(pose_data)
        else:
            pose_tensor = None

        input_tensor, target_tensor, pose_tensor = FLOAT_TRANSFORM(
            input_tensor, target_tensor, pose_tensor
        )
        if self.enable_augmentation:
            input_tensor, target_tensor, pose_tensor = AUGMENT_TRANSFORM(
                input_tensor, target_tensor, pose_tensor
            )
        if self.enable_normalization:
            input_tensor, target_tensor = NORMALIZE_TRANSFORM(
                input_tensor, target_tensor
            )

        if self.include_pose:
            input_tensor = tv_tensors.Image(
                torch.cat([input_tensor, pose_tensor], dim=0)
            )
        target_tensor = normalize_depth(target_tensor)

        return input_tensor, target_tensor


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = Cloth3dDataset(
        start_idx=0, end_idx=10, enable_augmentation=True, include_pose=True
    )
    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    input_batch, target_batch = next(iter(dataloader))
    input_tensor, target_tensor = input_batch[0], target_batch[0]
    print(input_tensor.shape, target_tensor.shape)

    img = input_tensor[:3].permute(1, 2, 0)

    cmap = plt.get_cmap("viridis")  # Get the viridis colormap directly
    depth = cmap(target_tensor.squeeze(0))[..., :3]
    pose = input_tensor[3:].permute(1, 2, 0)

    combined = np.concatenate([img, depth, pose], axis=1)
    plt.imshow(combined)
    plt.tight_layout()
    plt.show()
