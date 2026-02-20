import copy
import random
from typing import Union, Dict, List

import torch
import numpy as np
from mmengine.runner import Runner, BaseLoop
from mmseg.registry import RUNNERS, LOOPS

from .query_loop import QueryLoop


@RUNNERS.register_module()
class ActiveLearningRunner(Runner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._query_loop = self.cfg.query_cfg

    @property
    def query_loop(self):
        if isinstance(self._query_loop, QueryLoop) or self._query_loop is None:
            return self._query_loop
        else:
            self._query_loop = self.build_query_loop(self._query_loop)
            return self._query_loop

    @property
    def query_dataloader(self):
        return self.query_loop.dataloader

    def build_query_loop(self, loop: Union[QueryLoop, Dict]) -> QueryLoop:
        if isinstance(loop, QueryLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(f"query_loop should be a Loop object or dict, but got {loop}")

        loop_cfg = copy.deepcopy(loop)
        loop = LOOPS.build(
            loop_cfg,
            default_args=dict(
                runner=self,
                dataloader=self.cfg.query_dataloader
            )
        )

        return loop

    def query(self) -> None:
        if self._query_loop is None:
            raise RuntimeError("`self._query_loop` should not be None when calling query method.")

        self._query_loop = self.build_query_loop(self._query_loop)

        self.call_hook("before_run")
        self.load_or_resume()
        self._query_loop.run()
        self.call_hook("after_run")

    def set_randomness(self, seed, diff_rank_seed: bool = False, deterministic: bool = False) -> None:
        self._deterministic = deterministic
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = False
            torch.use_deterministic_algorithms(True, warn_only=True)
