import torch
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dummy_dataset() -> DummyDataset:
    return DummyDataset(
        [
            (torch.randn(3, 256, 256), torch.randn(1, 256, 256)),
            (torch.randn(3, 256, 256), torch.randn(1, 256, 256)),
        ]
    )


def get_dummy_dataloader(batch_size: int) -> DataLoader:
    return DataLoader(get_dummy_dataset(), batch_size=batch_size)
