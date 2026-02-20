import math
from typing import Literal, Union, Callable, Tuple
from time import time

import numpy as np
import torch


class KMeansClustering(object):
    """
    This class has been copied from the link below, and several parts have been modified.
    https://github.com/DeMoriarty/fast_pytorch_kmeans
    """

    def __init__(
            self,
            num_clusters: int,
            distance_func: Union[Literal["euclidean", "cosine", "custom"], Callable] = "euclidean",
            max_iter: int = 100,
            tolerance: float = 0.00001,
            minibatch_size: int = None,
            verbose: Literal[0, 1, 2] = 0,
    ):
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.minibatch_size = minibatch_size
        self.verbose = verbose
        self.centroids = None

        if isinstance(distance_func, str):
            self.mode = distance_func
            if distance_func == "euclidean":
                self._sim_func = KMeansClustering.euc_sim
            elif distance_func == "cosine":
                self._sim_func = KMeansClustering.cos_sim
            else:
                raise ValueError(distance_func)
        else:
            self.mode = "custom"
            self._sim_func = distance_func

        self._loop = False

    @staticmethod
    def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_norm = a.norm(dim=-1, keepdim=True)
        b_norm = b.norm(dim=-1, keepdim=True)
        a = a / (a_norm + 1e-8)
        b = b / (b_norm + 1e-8)
        return a @ b.transpose(-2, -1)

    @staticmethod
    def euc_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return 2 * a @ b.transpose(-2, -1) - (a ** 2).sum(dim=1)[..., :, None] - (b ** 2).sum(dim=1)[..., None, :]

    @staticmethod
    def remaining_memory():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated()

    def max_sim(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = a.device.type
        batch_size = a.shape[0]

        if device == "cpu":
            sim = self._sim_func(a, b)
            max_sim_v, max_sim_i = sim.max(dim=-1)
            return max_sim_v, max_sim_i
        else:
            expected_memory = a.shape[0] * a.shape[1] * b.shape[0] * 4
            if a.dtype == torch.half:
                expected_memory /= 2
            memory_ratio = math.ceil(expected_memory / self.remaining_memory())
            subbatch_size = math.ceil(batch_size / memory_ratio)

            msv, msi = [], []
            for i in range(memory_ratio):
                if i * subbatch_size >= batch_size:
                    continue

                sub_x = a[i * subbatch_size: (i + 1) * subbatch_size]
                sub_sim = self._sim_func(sub_x, b)
                sub_max_sim_v, sub_max_sim_i = sub_sim.max(dim=-1)
                del sub_sim
                msv.append(sub_max_sim_v)
                msi.append(sub_max_sim_i)

            if memory_ratio == 1:
                max_sim_v, max_sim_i = msv[0], msi[0]
            else:
                max_sim_v = torch.cat(msv, dim=0)
                max_sim_i = torch.cat(msi, dim=0)

            return max_sim_v, max_sim_i

    def fit_predict(self, embeddings: torch.Tensor, initial_centroids: torch.Tensor = None):
        batch_size, emb_dim = embeddings.shape
        device = embeddings.device.type
        start_time = time()

        if initial_centroids is None:
            rand_idx = np.random.choice(batch_size, size=[self.num_clusters], replace=False)
            self.centroids = embeddings[rand_idx]
        else:
            self.centroids = initial_centroids

        if self.verbose >= 1:
            print()

        num_points_in_clusters = torch.ones(self.num_clusters, device=device)
        closest_cluster_sim, closest_cluster_idx = None, None
        for i in range(self.max_iter):
            iter_time = time()

            if self.minibatch_size is not None:
                rand_idx = np.random.choice(batch_size, size=[self.minibatch_size], replace=False)
                x = embeddings[rand_idx]
            else:
                x = embeddings

            closest_cluster_sim, closest_cluster_idx = self.max_sim(x, self.centroids)
            matched_clusters, counts = closest_cluster_idx.unique(return_counts=True)

            c_grad = torch.zeros_like(self.centroids)
            if self._loop:
                for j, count in zip(matched_clusters, counts):
                    c_grad[j] = x[closest_cluster_idx == j].sum(dim=0) / count
            else:
                if self.minibatch_size is None:
                    expanded_closest = closest_cluster_idx[None].expand(self.num_clusters, -1)
                    mask = (expanded_closest == torch.arange(self.num_clusters, device=device)[:, None]).float()
                    c_grad = mask @ x / mask.sum(-1)[..., :, None]
                    c_grad[c_grad != c_grad] = 0  # remove NaNs
                else:
                    expanded_closest = closest_cluster_idx[None].expand(len(matched_clusters), -1)
                    mask = (expanded_closest == matched_clusters[:, None]).float()
                    c_grad[matched_clusters] = mask @ x / mask.sum(-1)[..., :, None]

            error = (c_grad - self.centroids).pow(2).sum()
            if self.minibatch_size is not None:
                lr = 1 / num_points_in_clusters[:, None] * 0.9 + 0.1
            else:
                lr = 1

            num_points_in_clusters[matched_clusters] += counts
            self.centroids = self.centroids * (1 - lr) + c_grad * lr

            if self.verbose >= 2:
                elapsed = round(time() - iter_time, 4)
                print("iter:", i, "error:", error.item(), "time spent:", elapsed)

            if error <= self.tolerance:
                break

        if self.verbose >= 1:
            elapsed = round(time() - start_time, 4)
            print(f"{i + 1} iterations ({elapsed}s) to cluster {batch_size} items into {self.num_clusters} clusters")

        return closest_cluster_sim, closest_cluster_idx

    def predict(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.max_sim(embeddings, self.centroids)

    def fit(self, embeddings: torch.Tensor, initial_centroids: torch.Tensor = None):
        self.fit_predict(embeddings, initial_centroids)
