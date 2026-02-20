from typing import List

import torch

from .registry import SELECTORS
from .base_selector import BaseSelector


@SELECTORS.register_module()
class TopK(BaseSelector):
    def __init__(self):
        super().__init__()

    def select(self, num_samples: int, values: torch.Tensor, unlabelled_idx: torch.LongTensor) -> List[int]:
        unlabelled_vals = values[unlabelled_idx][:, 0].cpu()  # MPS device does not support topk when k >= 16
        _, sorted_idx = torch.topk(unlabelled_vals, k=num_samples, dim=0)
        selected_idx = [int(unlabelled_idx[i]) for i in sorted_idx]
        return selected_idx
