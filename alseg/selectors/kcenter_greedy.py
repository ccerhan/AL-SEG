from typing import List

import torch
from mmengine.device import get_device

from .registry import SELECTORS
from .base_selector import BaseSelector
from .distance_metrics import create_distance_metric, EuclideanDistance


@SELECTORS.register_module()
class KCenterGreedy(BaseSelector):
    def __init__(self, distance_metric: str = EuclideanDistance.NAME):
        super().__init__()
        self.distance_metric = create_distance_metric(distance_metric, device=get_device())
        self.weights = None
        self.dist_mat = None
        self.tau = 1

    def select(self, num_samples: int, values: torch.Tensor, unlabelled_idx: torch.LongTensor) -> List[int]:
        embeddings = values
        dataset_size = embeddings.shape[0]

        labelled_mask = torch.full((dataset_size,), True)
        labelled_mask[unlabelled_idx] = False

        # compute distance matrix of all embeddings
        dist_mat = self.distance_metric.compute_distance_matrix(embeddings)

        if self.weights is not None:
            dist_mat = self.distance_metric.apply_weights(dist_mat, self.weights, self.tau).T

        mat = dist_mat[~labelled_mask, :][:, labelled_mask]  # [ unlabelled_idx, labelled_idx ]

        selected_idx = torch.zeros(num_samples).to(torch.long)
        for i in range(num_samples):
            mat_idx = mat.min(dim=1)[0].argmax()  # selection criteria, returns unlabelled idx wrt mat
            selected_idx[i] = torch.arange(dataset_size)[~labelled_mask][mat_idx]  # map selected idx wrt dataset

            # mark the selected idx as labelled
            labelled_mask[selected_idx[i]] = True

            # remove selected idx from unlabelled
            preserve_idx = torch.LongTensor([j for j in range(mat.shape[0]) if j != mat_idx])
            mat = mat[preserve_idx]

            # append selected idx as labelled to mat
            labelled_col = dist_mat[~labelled_mask, selected_idx[i]][:, None]
            mat = torch.cat([mat, labelled_col], dim=1)

        selected_idx = selected_idx.cpu().tolist()

        self.dist_mat = dist_mat.T
        return selected_idx
