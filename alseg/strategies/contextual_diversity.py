from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader

from mmengine.registry import LOOPS
from mmengine.device import get_device

from ..runners import QueryLoop
from ..selectors import KCenterGreedy


@LOOPS.register_module()
class ContextualDiversity(QueryLoop):
    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            num_samples: int,
            current_split_file: str = None,
            output_split_file: str = None,
            fp16: bool = False,
            ignore_type: str = "none"
    ) -> None:
        super().__init__(runner, dataloader, num_samples, current_split_file, output_split_file, fp16, ignore_type)
        num_cls = self.runner.model.num_classes
        max_prob = torch.ones(num_cls) / num_cls
        self._max_score = float((-max_prob * torch.log(max_prob)).sum())

    @torch.no_grad()
    def compute_iter(self, idx: int, data_batch: Dict) -> List[torch.Tensor]:
        seg_results = super().compute_iter(idx, data_batch)
        seg_logits = torch.stack([r.seg_logits.data for r in seg_results])
        probs = torch.softmax(seg_logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        if self.ignore_type == "index" and self.query_split.dataset.ignore_index is not None:
            labels = torch.vstack([r.gt_sem_seg.data for r in seg_results]).to(get_device())
            ignore_mask = (labels == self.query_split.dataset.ignore_index)
            preds[ignore_mask] = self.query_split.dataset.ignore_index

        entropy_map = torch.nansum(-probs * torch.log(probs), dim=1)
        entropy_map /= self._max_score  # normalize to [0, 1]

        batch_size = len(seg_results)
        num_cls = self.runner.model.num_classes
        features = torch.full((batch_size, num_cls, num_cls), torch.finfo(torch.float32).eps).to(get_device())
        for batch_idx in range(batch_size):
            for cls_idx in range(num_cls):
                cls_mask = preds[batch_idx] == cls_idx
                if cls_mask.sum() > 0:
                    # get softmax vectors for the pixels that are classified as 'cls_idx' - returns shape [num_cls, :]
                    cls_probs = probs[batch_idx, :, cls_mask]
                    # get the entropy of the softmax vectors - returns shape [:, 1]
                    cls_weights = entropy_map[batch_idx, cls_mask].view(-1, 1)
                    # take weighted average of each class softmax score with entropy weights
                    weighted_avg = (cls_probs @ cls_weights) / torch.sum(cls_weights)
                    features[batch_idx, cls_idx] = weighted_avg.flatten()

        values = features.view(features.shape[0], -1)
        return values

    def select(self, num_samples: int, values: torch.Tensor, unlabelled_idx: torch.LongTensor) -> List[int]:
        kc = KCenterGreedy(distance_metric="kl_div")
        selected_idx = kc.select(num_samples, values, unlabelled_idx)
        return selected_idx
