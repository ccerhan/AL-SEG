from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mmengine.registry import LOOPS
from mmseg.models.decode_heads import ASPPHead, SegformerHead
from mmseg.models.utils import resize

from ..runners import QueryLoop
from ..selectors import KCenterGreedy


@LOOPS.register_module()
class CoreSet(QueryLoop):
    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            num_samples: int,
            current_split_file: str = None,
            output_split_file: str = None,
            fp16: bool = False,
            embedding_type: str = "bottleneck",  # 'penultimate' or 'bottleneck'
            tau: float = 1,
    ) -> None:
        super().__init__(runner, dataloader, num_samples, current_split_file, output_split_file, fp16)
        self.embedding_type = embedding_type
        self.tau = tau

    @torch.no_grad()
    def compute_iter(self, idx: int, data_batch: Dict) -> torch.Tensor:
        data_batch = self.runner.model.data_preprocessor(data_batch, False)
        x = self.runner.model.extract_feat(data_batch["inputs"])

        if self.embedding_type == "penultimate":
            features = self.get_penultimate_features(x)
        elif self.embedding_type == "bottleneck":
            features = x[-1]
        else:
            raise ValueError("Invalid embedding_type")

        values = F.avg_pool2d(features, kernel_size=features.shape[2:]).view(features.shape[0], -1)

        if self.tau < 1:
            # use entropy for uncertainty to use barycenter distance
            batch_img_metas = [data_sample.metainfo for data_sample in data_batch["data_samples"]]
            seg_logits = self.runner.model.decode_head.predict(x, batch_img_metas, self.runner.model.test_cfg)
            probs = torch.softmax(seg_logits, dim=1)
            entropy_map = torch.nansum(-probs * torch.log(probs), dim=1)
            weights = entropy_map.mean(dim=(1, 2)).unsqueeze(1)
            values = torch.hstack([weights, values])

        return values

    def get_penultimate_features(self, x: List[torch.Tensor]):
        decode_head = self.runner.model.decode_head

        if isinstance(decode_head, ASPPHead):
            # copied from ASPPHead forward function
            features = decode_head._forward_feature(x)

        elif isinstance(decode_head, SegformerHead):
            # copied from SegformerHead forward function
            inputs = decode_head._transform_inputs(x)
            outs = []
            for idx in range(len(inputs)):
                x = inputs[idx]
                conv = decode_head.convs[idx]
                outs.append(
                    resize(
                        input=conv(x),
                        size=inputs[0].shape[2:],
                        mode=decode_head.interpolate_mode,
                        align_corners=decode_head.align_corners))

            features = decode_head.fusion_conv(torch.cat(outs, dim=1))   # 256, 120, 120

        else:
            raise ValueError("Unsupported DecodeHead")

        return features

    def select(self, num_samples: int, values: torch.Tensor, unlabelled_idx: torch.LongTensor) -> List[int]:
        kc = KCenterGreedy(distance_metric="euclidean")
        embeddings = values
        if self.tau < 1:
            kc.tau = self.tau
            kc.weights = values[:, 0]
            embeddings = values[:, 1:]
        selected_idx = kc.select(num_samples, embeddings, unlabelled_idx)
        return selected_idx
