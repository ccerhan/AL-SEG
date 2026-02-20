from typing import List, Literal

import torch
from mmengine.device import get_device

from .registry import SELECTORS
from .base_selector import BaseSelector
from ..helpers import KMeansClustering


@SELECTORS.register_module()
class KMeansCentroid(BaseSelector):
    def __init__(self, distance_metric: Literal["euclidean", "cosine"] = "euclidean"):
        super().__init__()
        self.distance_metric = distance_metric
        if self.distance_metric not in ["euclidean", "cosine"]:
            raise ValueError("Invalid 'distance_metric'")

    def select(self, num_samples: int, values: torch.Tensor, unlabelled_idx: torch.LongTensor) -> List[int]:
        embeddings = values
        unlabelled_embeddings = embeddings[unlabelled_idx]
        unlabelled_size = unlabelled_embeddings.shape[0]

        kmeans = KMeansClustering(num_clusters=num_samples, distance_func=self.distance_metric)
        cluster_sim, cluster_idx = kmeans.fit_predict(unlabelled_embeddings)

        selected_idx = torch.zeros(num_samples).to(torch.long)
        for i in range(num_samples):
            # select a sample that is closest to the associated cluster centroid
            emb_idx = torch.arange(unlabelled_size)
            cls_idx = cluster_sim[cluster_idx == i].argmax()
            selected_idx[i] = emb_idx[cluster_idx == i][cls_idx]

        idx_map = torch.arange(embeddings.shape[0])[unlabelled_idx]
        selected_idx = idx_map[selected_idx].tolist()
        return selected_idx
