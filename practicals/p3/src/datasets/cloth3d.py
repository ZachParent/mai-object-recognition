from typing import Tuple

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
        v2.RandomRotation([-10, 10]),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomErasing(p=0.2),
        v2.Normalize(mean=MEAN, std=STD),
    ]
)


class Cloth3dDataset(Dataset):
    def __init__(self, enable_augmentation=False):
        self.enable_augmentation = enable_augmentation
        self.image_paths = sorted(PREPROCESSED_DATA_DIR.glob("**/rgb/*.png"))
        self.depth_paths = sorted(PREPROCESSED_DATA_DIR.glob("**/depth_vis/*.png"))
        assert len(self.image_paths) == len(self.depth_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[tv_tensors.Image, tv_tensors.Mask]:
        input_data = torch.from_numpy(np.array(Image.open(self.image_paths[idx])))
        target_data = torch.from_numpy(np.array(Image.open(self.depth_paths[idx])))

        # move channel to first dimension, and drop alpha channel
        input_data = einops.rearrange(input_data, "h w c -> c h w")[:3]

        # Wrap raw tensors into tv_tensors.Image and tv_tensors.Mask
        # This allows v2 transforms to correctly identify and process them.
        input_tensor = tv_tensors.Image(input_data)
        target_tensor = tv_tensors.Mask(target_data)

        if self.enable_augmentation:
            # Apply the unified augmentation transform to the pair
            input_tensor, target_tensor = AUGMENT_TRANSFORM(input_tensor, target_tensor)
        else:
            # Apply default transform to input; target mask usually doesn't get image normalization
            input_tensor = DEFAULT_INPUT_TRANSFORM(input_tensor)
        return input_tensor, target_tensor


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = Cloth3dDataset(enable_augmentation=True)
    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    input_batch, target_batch = next(iter(dataloader))
    input_tensor, target_tensor = input_batch[0], target_batch[0]
    print(input_tensor.shape, target_tensor.shape)
    plt.imshow(input_tensor.permute(1, 2, 0))
    plt.show()
    plt.imshow(target_tensor)
    plt.show()
