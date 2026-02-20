from typing import List

import torch


class BaseSelector(object):
    def __init__(self):
        pass

    def select(self, num_samples: int, values: torch.Tensor, unlabelled_idx: torch.LongTensor) -> List[int]:
        raise NotImplementedError
