from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader

from mmengine.registry import LOOPS
from mmengine.device import get_device

from ..runners import QueryLoop
from ..helpers import UncertaintyMap
from ..selectors import TopK, RandomBatch


@LOOPS.register_module()
class Entropy(QueryLoop):
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

        num_cls = self.runner.model.num_classes
        max_prob = torch.ones(num_cls) / num_cls
        self._max_score = float((-max_prob * torch.log(max_prob)).sum())

    @torch.no_grad()
    def compute_iter(self, idx: int, data_batch: Dict) -> torch.Tensor:
        seg_results = super().compute_iter(idx, data_batch)
        seg_logits = torch.stack([r.seg_logits.data for r in seg_results])
        probs = torch.softmax(seg_logits, dim=1)

        ignore_mask, valid_mask = None, None
        if self.ignore_type == "index" and self.query_split.dataset.ignore_index is not None:
            labels = torch.vstack([r.gt_sem_seg.data for r in seg_results]).to(get_device())
            ignore_mask = (labels == self.query_split.dataset.ignore_index)
            valid_mask = ~ignore_mask

        um = UncertaintyMap(valid_mask, device=get_device())

        entropy_map = torch.nansum(-probs * torch.log(probs), dim=1)
        entropy_map /= self._max_score  # normalize to [0, 1]

        values = um.compute_mean_uncertainty(entropy_map)
        return values

    def select(self, num_samples: int, values: torch.Tensor, unlabelled_idx: torch.LongTensor) -> List[int]:
        if self.pool_size is None:
            selected_idx = TopK().select(num_samples, values, unlabelled_idx)
        else:
            selected_idx = RandomBatch(self.pool_size).select(num_samples, values, unlabelled_idx)
        return selected_idx
