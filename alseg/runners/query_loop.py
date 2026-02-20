import os
import json
import logging
import os.path as osp
from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from mmengine.device import get_device
from mmengine.runner import BaseLoop, autocast
from mmengine.logging import print_log

from .query_split import QuerySplit
from ..utils.constants import VALUES_FILE_NPY, META_FILE_JSON


class QueryLoop(BaseLoop):
    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            num_samples: int,
            current_split_file: str = None,
            output_split_file: str = None,
            fp16: bool = False,
            ignore_type: str = "none",  # "none" or "index"
    ) -> None:
        super().__init__(runner, dataloader)
        self.num_samples = num_samples
        self.current_split_file = current_split_file
        self.output_split_file = output_split_file
        self.fp16 = fp16
        self.ignore_type = ignore_type

        self._DEBUG = os.environ.get("DEBUG", False)

        self.query_split = QuerySplit(self.dataloader.dataset, self.current_split_file)
        self.meta = {}  # to store intermediate meta data

        if hasattr(self.dataloader.dataset, "metainfo"):
            self.runner.visualizer.dataset_meta = self.dataloader.dataset.metainfo
        else:
            print_log(
                f"Dataset {self.dataloader.dataset.__class__.__name__} has no "
                "metainfo. ``dataset_meta`` in evaluator, metric and "
                "visualizer will be None.",
                logger="current",
                level=logging.WARNING)

    def run(self) -> None:
        # compute
        outputs = self.compute()
        if len(outputs) > 0:
            outputs = torch.stack(outputs)
        else:
            outputs = None

        if outputs is not None:
            values_file_path = osp.join(self.runner.work_dir, VALUES_FILE_NPY)
            np.save(values_file_path, outputs.detach().cpu().numpy())

        # select
        if self.num_samples < len(self.query_split.unlabelled_idx):
            unlabelled_idx = torch.LongTensor(self.query_split.unlabelled_idx).to(get_device())
            selected_idx = self.select(self.num_samples, outputs, unlabelled_idx)
        else:
            selected_idx = self.query_split.unlabelled_idx

        self.meta["dataset"] = dict(
            labelled_idx=self.query_split.labelled_idx,
            unlabelled_idx=self.query_split.unlabelled_idx,
            selected_idx=[int(i) for i in selected_idx],
            remaining_idx=sorted(list(set(self.query_split.unlabelled_idx).difference(selected_idx))),
        )

        meta_file_path = osp.join(self.runner.work_dir, META_FILE_JSON)
        with open(meta_file_path, "w") as fp:
            json.dump(self.meta, fp)

        self.runner.logger.info(f"Selected set length: {len(selected_idx)}")
        self.runner.logger.info(f"Selected set indexes:\n{selected_idx}")

        # merge
        self.query_split.save(selected_idx, self.output_split_file)

    def compute(self) -> List[torch.Tensor]:
        self.runner.call_hook("before_test_epoch")
        outputs = []
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.runner.call_hook("before_test_iter", batch_idx=idx, data_batch=data_batch)
            with autocast(enabled=self.fp16):
                output = self.compute_iter(idx, data_batch)
            if output is None:
                break
            outputs.extend(output)
            self.runner.call_hook("after_test_iter", batch_idx=idx, data_batch=data_batch, outputs=[])
        self.runner.call_hook("after_test_epoch")
        return outputs

    @torch.no_grad()
    def compute_iter(self, idx: int, data_batch: Dict, postprocess: bool = True) -> torch.Tensor:
        data_batch = self.runner.model.data_preprocessor(data_batch, False)
        inputs, data_samples = data_batch["inputs"], data_batch["data_samples"]
        batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        seg_logits = self.runner.model.inference(inputs, batch_img_metas)
        if not postprocess:
            return seg_logits
        seg_results = self.runner.model.postprocess_result(seg_logits, data_samples)
        return seg_results

    def select(self, num_samples: int, values: torch.Tensor, unlabelled_idx: torch.LongTensor) -> List[int]:
        raise NotImplementedError
