from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from scipy import stats

from mmengine.registry import LOOPS
from mmengine.device import get_device

from ..runners import QueryLoop


@LOOPS.register_module()
class Badge(QueryLoop):
    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            num_samples: int,
            current_split_file: str = None,
            output_split_file: str = None,
            fp16: bool = False,
            embedding_size: int = 400,
    ) -> None:
        super().__init__(runner, dataloader, num_samples, current_split_file, output_split_file, fp16)
        self.embedding_size = embedding_size
        self.avg_pool_kernel = None

    @torch.no_grad()
    def compute_iter(self, idx: int, data_batch: Dict) -> torch.Tensor:
        seg_logits = super().compute_iter(idx, data_batch, postprocess=False)
        probs = torch.softmax(seg_logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        with torch.enable_grad():
            seg_logits.requires_grad = True
            loss = F.cross_entropy(seg_logits, preds, reduction="sum")
            grads = torch.autograd.grad(loss, seg_logits)[0]

        if self.avg_pool_kernel is None:
            # determine the avg pool kernel size for which the embedding size is close to the desired value
            max_size = int(min(*grads.size()[2:]))
            embedding_size = np.zeros(max_size, dtype=int)
            for k in range(1, max_size + 1):
                embedding_size[max_size - k] = F.avg_pool2d(grads[0], kernel_size=(k, k)).numel()
            k = np.argmin(np.abs(embedding_size - self.embedding_size))
            kernel_size = max_size - k
            self.avg_pool_kernel = (kernel_size, kernel_size)
            self.embedding_size = int(embedding_size[k])

        values = F.avg_pool2d(grads, kernel_size=self.avg_pool_kernel).view(grads.shape[0], -1)
        return values

    def select(self, num_samples: int, values: torch.Tensor, unlabelled_idx: torch.LongTensor) -> List[int]:
        dataset_size = values.shape[0]
        embeddings = values[unlabelled_idx].cpu().numpy()

        selected_idx = self._init_centers(embeddings, num_samples, get_device())

        # map unlabelled indexes to dataset indexes
        selected_idx = torch.LongTensor(selected_idx)
        idx_map = torch.arange(dataset_size)[unlabelled_idx.cpu()]
        selected_idx = idx_map[selected_idx].tolist()
        return selected_idx

    @staticmethod
    def _init_centers(X, K, device):
        # https://github.com/decile-team/distil/blob/main/distil/active_learning_strategies/badge.py
        pdist = nn.PairwiseDistance(p=2)
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pdist(torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device))
                D2 = torch.flatten(D2)
                D2 = D2.cpu().numpy().astype(float)
            else:
                newD = pdist(torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device))
                newD = torch.flatten(newD)
                newD = newD.cpu().numpy().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]

            if sum(D2) == 0.0:
                import pdb
                pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll
