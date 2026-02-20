import os.path as osp
import copy

from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.registry import RUNNERS


def train_model(
        cfg: Config,
        split_file: str = None,
        options: dict = None,
):
    cfg = copy.deepcopy(cfg)
    if options is not None:
        cfg.merge_from_dict(options)

    if split_file is not None:
        if not osp.isabs(split_file):
            split_file = osp.join(cfg.work_dir, split_file)
        cfg.train_dataloader.dataset.ann_file = split_file

    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.train()
