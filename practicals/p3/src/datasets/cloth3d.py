import re
from pathlib import Path
from typing import List, Optional, Tuple

import einops
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from ..config import PREPROCESSED_DATA_DIR

# @TODO: Calculate dataset means and stds for actual dataset
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Non-normalized input transform
NON_NORMALIZED_INPUT_TRANSFORM = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
    ]
)

# Transform for non-augmented input image
DEFAULT_INPUT_TRANSFORM = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
    ]
)

# Transform for unified augmentation of input image and target mask
AUGMENT_TRANSFORM = v2.Compose(
    [
        v2.RandomRotation([-5, 5]),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        v2.Normalize(mean=MEAN, std=STD),
    ]
)


def get_image_and_depth_paths(
    start_idx: int, end_idx: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    # Get all parent folders and sort them
    parent_folders = sorted(PREPROCESSED_DATA_DIR.glob("0*"))
    if end_idx is None:
        end_idx = len(parent_folders)

    selected_folders = parent_folders[start_idx:end_idx]

    # Get image and depth paths only from selected folders
    image_paths = []
    depth_paths = []
    for folder in selected_folders:
        image_paths.extend(sorted(folder.glob("rgb/*.png")))
        depth_paths.extend(sorted(folder.glob("depth_vis/*.png")))

    assert len(image_paths) == len(depth_paths)
    return image_paths, depth_paths


class Cloth3dDataset(Dataset):
    def __init__(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        enable_normalization: bool = True,
        enable_augmentation: bool = False,
    ):
        self.enable_normalization = enable_normalization
        self.enable_augmentation = enable_augmentation

        # Get parent folders to calculate num_videos and store video_ids
        parent_folders = sorted(PREPROCESSED_DATA_DIR.glob("0*"))
        if end_idx is None:
            end_idx = len(parent_folders)
        self.num_videos = end_idx - start_idx
        self.video_ids = [
            int(folder.name) for folder in parent_folders[start_idx:end_idx]
        ]

        self.image_paths, self.depth_paths = get_image_and_depth_paths(
            start_idx, end_idx
        )

    def _get_image_and_depth_paths(self, idx: int) -> Tuple[str, str]:
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
        target_data = (
            torch.from_numpy(np.array(Image.open(self.depth_paths[idx]))).float()
            / 255.0
        )

        # move channel to first dimension, and drop alpha channel
        input_data = einops.rearrange(input_data, "h w c -> c h w")[:3]
        target_data = einops.rearrange(target_data, "h w -> () h w")

        # Wrap raw tensors into tv_tensors.Image and tv_tensors.Mask
        # This allows v2 transforms to correctly identify and process them.
        input_tensor = tv_tensors.Image(input_data)
        target_tensor = tv_tensors.Mask(target_data)

        if self.enable_augmentation:
            # Apply the unified augmentation transform to the pair
            input_tensor, target_tensor = AUGMENT_TRANSFORM(input_tensor, target_tensor)
        elif self.enable_normalization:
            # Apply default transform to input; target mask usually doesn't get image normalization
            input_tensor = DEFAULT_INPUT_TRANSFORM(input_tensor)
        else:
            input_tensor = NON_NORMALIZED_INPUT_TRANSFORM(input_tensor)

        return input_tensor, target_tensor


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = Cloth3dDataset(start_idx=0, end_idx=10, enable_augmentation=True)
    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    input_batch, target_batch = next(iter(dataloader))
    input_tensor, target_tensor = input_batch[0], target_batch[0]
    print(input_tensor.shape, target_tensor.shape)
    plt.imshow(input_tensor.permute(1, 2, 0))
    plt.show()
    plt.imshow(target_tensor.squeeze(0))
    plt.colorbar()
    plt.show()
