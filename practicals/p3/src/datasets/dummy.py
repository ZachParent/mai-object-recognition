import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

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
        v2.RandomRotation(10),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomErasing(p=0.2),
        v2.Normalize(mean=MEAN, std=STD),
    ]
)


class DummyDataset(Dataset):
    def __init__(self, data, augmentation=False):
        self.data = data
        self.augmentation = augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data, target_data = self.data[idx]

        # Wrap raw tensors into tv_tensors.Image and tv_tensors.Mask
        # This allows v2 transforms to correctly identify and process them.
        input_tensor = tv_tensors.Image(input_data)
        target_tensor = tv_tensors.Mask(target_data)

        if self.augmentation:
            # Apply the unified augmentation transform to the pair
            input_tensor, target_tensor = AUGMENT_TRANSFORM(input_tensor, target_tensor)
        else:
            # Apply default transform to input; target mask usually doesn't get image normalization
            input_tensor = DEFAULT_INPUT_TRANSFORM(input_tensor)
        return input_tensor, target_tensor


def get_dummy_dataset() -> DummyDataset:
    return DummyDataset(
        [
            (torch.randn(3, 256, 256), torch.randn(1, 256, 256)),
            (torch.randn(3, 256, 256), torch.randn(1, 256, 256)),
        ]
    )


def get_dummy_dataloader(batch_size: int) -> DataLoader:
    return DataLoader(get_dummy_dataset(), batch_size=batch_size)
