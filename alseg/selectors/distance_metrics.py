from typing import Tuple

import torch


class DistanceMetric(object):
    NAME = None

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __str__(self) -> str:
        return self.NAME

    def compute_distance_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply_weights(self, dist_mat: torch.Tensor, weights: torch.Tensor, tau: float = 1.0):
        num_embeddings = dist_mat.shape[0]
        eps = torch.finfo(torch.float32).eps  # numerical stability

        # multiply by weights like in the two-body problem
        w_j = torch.tile(weights, (num_embeddings,)).view(num_embeddings, num_embeddings)
        w_i = torch.tile(weights.unsqueeze(dim=1), (num_embeddings,)).view(num_embeddings, num_embeddings)
        w = w_j / (w_i + w_j + eps)

        return torch.exp(tau * torch.log(dist_mat + eps) + (1 - tau) * torch.log(w))


class EuclideanDistance(DistanceMetric):
    NAME = "euclidean"

    def __init__(self, device: str = "cpu"):
        super().__init__(device)

    def compute_distance_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        num_embeddings = embeddings.shape[0]
        embeddings = embeddings.view(num_embeddings, -1)
        dist_mat = torch.matmul(embeddings, embeddings.T)
        sq = dist_mat.diagonal().detach().clone().view(num_embeddings, 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.T
        dist_mat[dist_mat < 0] = 0  # numerical stability
        dist_mat = torch.sqrt(dist_mat)
        return dist_mat


class BarycenterDistance(EuclideanDistance):
    NAME = "barycenter"

    def __init__(self, device: str, tau: float):
        super().__init__(device)
        if tau < 0 or tau > 1:
            raise ValueError("tau must be in the range [0, 1]")
        self.tau = tau

    def compute_distance_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        return super().compute_distance_matrix(embeddings)


class KLDivergence(DistanceMetric):
    NAME = "kl_div"

    def __init__(self, device: str = "cpu"):
        super().__init__(device)

    def compute_distance_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        num_embeddings = embeddings.shape[0]
        embeddings = embeddings.view(num_embeddings, -1)

        eps = torch.finfo(torch.float32).eps

        # # Old implementation - takes too much time due to the for loops
        # dist_mat = torch.zeros((num_embeddings, num_embeddings), dtype=torch.float32).to(self.device)
        # for i in range(num_embeddings):
        #     for j in range(i):
        #         e1, e2 = embeddings[i], embeddings[j]
        #         kl1 = torch.sum(e1 * torch.log(e1 / (e2 + eps) + eps), dim=0)
        #         kl2 = torch.sum(e2 * torch.log(e2 / (e1 + eps) + eps), dim=0)
        #         sum_dist = torch.sum(kl1 + kl2) / 2
        #         dist_mat[i, j] = dist_mat[j, i] = sum_dist

        # Efficient implementation computed in chunks
        dist_mat = torch.zeros(num_embeddings * num_embeddings, dtype=torch.float32).to(self.device)

        idx_pairs = [[i, j] for i in range(num_embeddings) for j in range(num_embeddings)]
        idx_pairs = torch.as_tensor(idx_pairs)

        chunk_size = 100000
        total_chunks = round(len(dist_mat) / chunk_size + 0.5)
        for c in range(total_chunks):
            c_start = c * chunk_size
            c_end = min((c + 1) * chunk_size, len(dist_mat))
            e1 = embeddings[idx_pairs[c_start: c_end, 0]]
            e2 = embeddings[idx_pairs[c_start: c_end, 1]]
            kl1 = torch.sum(e1 * torch.log(e1 / (e2 + eps) + eps), dim=1)
            kl2 = torch.sum(e2 * torch.log(e2 / (e1 + eps) + eps), dim=1)
            dist_mat[c_start: c_end] = (kl1 + kl2) / 2
        dist_mat = dist_mat.reshape(num_embeddings, num_embeddings)
        return dist_mat


def create_distance_metric(distance_metric: str, device: str = "cpu", **kwargs) -> DistanceMetric:
    kwargs["device"] = device
    if distance_metric is not None and distance_metric.lower() == EuclideanDistance.NAME:
        return EuclideanDistance(**kwargs)
    if distance_metric is not None and distance_metric.lower() == KLDivergence.NAME:
        return KLDivergence(**kwargs)
    raise ValueError("Invalid distance metric")
