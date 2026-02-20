from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader
from mmengine.registry import LOOPS

from ..runners import QueryLoop
from ..selectors import RandomBatch


@LOOPS.register_module()
class Random(QueryLoop):
    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            num_samples: int,
            current_split_file: str = None,
            output_split_file: str = None,
            fp16: bool = False,
            **kwargs
    ) -> None:
        super().__init__(runner, dataloader, num_samples, current_split_file, output_split_file, fp16)

    def compute(self) -> List[torch.Tensor]:
        return []

    def select(self, num_samples: int, values: torch.Tensor, unlabelled_idx: torch.LongTensor) -> List[int]:
        selected_idx = RandomBatch(pool_size=None).select(num_samples, values, unlabelled_idx)
        return selected_idx
