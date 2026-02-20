from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader

from mmengine.registry import LOOPS

from ..runners import QueryLoop
from ..selectors import TopK, RandomBatch


@LOOPS.register_module()
class LeastConfidence(QueryLoop):
    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            num_samples: int,
            current_split_file: str = None,
            output_split_file: str = None,
            fp16: bool = False,
            ignore_type: str = "none",
            pool_size: int = None,
    ) -> None:
        super().__init__(runner, dataloader, num_samples, current_split_file, output_split_file, fp16, ignore_type)
        self.pool_size = pool_size

    @torch.no_grad()
    def compute_iter(self, idx: int, data_batch: Dict) -> torch.Tensor:
        seg_results = super().compute_iter(idx, data_batch)
        seg_logits = torch.stack([r.seg_logits.data for r in seg_results])
        probs = torch.softmax(seg_logits, dim=1)
        probs_max, _ = probs.max(dim=1)
        uncert_imgs = 1 - probs_max
        return uncert_imgs.mean(dim=(1, 2)).unsqueeze(1)

    def select(self, num_samples: int, values: torch.Tensor, unlabelled_idx: torch.LongTensor) -> List[int]:
        if self.pool_size is None:
            selected_idx = TopK().select(num_samples, values, unlabelled_idx)
        else:
            selected_idx = RandomBatch(self.pool_size).select(num_samples, values, unlabelled_idx)
        return selected_idx
