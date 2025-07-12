from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F


class WellLogDataset(data.Dataset):
    """Loads 1-D well log samples stored as ``.npy`` files."""

    def __init__(self, root: str, length: int) -> None:
        self.root = Path(root)
        self.paths = sorted(self.root.glob('*.npy'))
        if not self.paths:
            raise RuntimeError(f'no .npy files found in {self.root}')
        self.length = length

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.paths)

    def __getitem__(self, index: int):  # type: ignore[override]
        arr = np.load(self.paths[index]).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        tensor = torch.from_numpy(arr)
        if tensor.shape[-1] > self.length:
            start = torch.randint(0, tensor.shape[-1] - self.length + 1, ()).item()
            tensor = tensor[..., start:start + self.length]
        elif tensor.shape[-1] < self.length:
            pad = self.length - tensor.shape[-1]
            tensor = F.pad(tensor, (0, pad))
        tensor = tensor.unsqueeze(-1)
        aug_cond = torch.zeros(9, dtype=torch.float32)
        return tensor, tensor.clone(), aug_cond


def get_dataset(config: dict, transform: Optional[callable] = None) -> WellLogDataset:
    """Factory used by ``train.py`` to construct the dataset."""
    root = config['root']
    length = config.get('length', 256)
    return WellLogDataset(root, length)
