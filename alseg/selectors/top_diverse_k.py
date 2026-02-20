from typing import List

import torch
from mmengine.device import get_device

from .registry import SELECTORS
from .base_selector import BaseSelector
from .distance_metrics import create_distance_metric, EuclideanDistance


@SELECTORS.register_module()
class TopDiverseK(BaseSelector):
    def __init__(self, distance_metric: str = EuclideanDistance.NAME, tau: float = 1.0):
        super().__init__()
        self.distance_metric = create_distance_metric(distance_metric, device=get_device())
        self.tau = tau
        self.dist_mat = None

    def select(self, num_samples: int, values: torch.Tensor, unlabelled_idx: torch.LongTensor) -> List[int]:
        weights = values[:, 0]
        embeddings = values[:, 1:]
        dataset_size = values.shape[0]

        labelled_mask = torch.full((dataset_size,), True)
        labelled_mask[unlabelled_idx] = False

        # compute distance matrix of embeddings
        dist_mat = self.distance_metric.compute_distance_matrix(embeddings)
        dist_mat = self.distance_metric.apply_weights(dist_mat, weights, self.tau)

        selected_idx = []
        while len(selected_idx) < min(num_samples, len(unlabelled_idx)):
            traversed_idx = torch.nonzero(labelled_mask).view(-1)
            if len(traversed_idx) == 0:
                idx = torch.argmax(weights)
            else:
                dist = dist_mat[traversed_idx]
                dist_func = dist.mean(dim=0) + dist.min(dim=0)[0]
                dist_func[traversed_idx] = -1  # prevents selecting traversed samples with argmax
                idx = torch.argmax(dist_func)
            labelled_mask[idx] = True
            selected_idx.append(int(idx))

        self.dist_mat = dist_mat
        return selected_idx
