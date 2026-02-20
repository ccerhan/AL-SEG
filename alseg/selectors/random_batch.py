from typing import List

import numpy as np
import torch

from .registry import SELECTORS
from .base_selector import BaseSelector


@SELECTORS.register_module()
class RandomBatch(BaseSelector):
    def __init__(self, pool_size: int = None):
        super().__init__()
        self.pool_size = pool_size

    def select(self, num_samples: int, values: torch.Tensor, unlabelled_idx: torch.LongTensor) -> List[int]:
        unlabelled_idx = unlabelled_idx.cpu().tolist()

        if self.pool_size is None:
            selected_idx = np.random.choice(unlabelled_idx, num_samples, replace=False)
            return list(selected_idx)

        max_value = -1
        selected_idx = None
        values = values.cpu().numpy().reshape(-1)
        for _ in range(min(self.pool_size, len(unlabelled_idx))):
            sampled_idx = np.random.choice(unlabelled_idx, num_samples, replace=False)
            value = np.mean(values[sampled_idx])
            if value > max_value:
                max_value = value
                selected_idx = sampled_idx

        return selected_idx
