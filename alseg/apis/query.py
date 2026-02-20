import os.path as osp
import copy

from mmengine.config import Config
from mmseg.registry import RUNNERS


def query_samples(
        cfg: Config,
        num_samples: int,
        output_split_file: str,
        current_split_file: str = None,
        checkpoint_file: str = None,
        options: dict = None,
):
    cfg = copy.deepcopy(cfg)
    if options is not None:
        cfg.merge_from_dict(options)

    cfg.query_cfg.num_samples = num_samples

    if current_split_file is not None:
        if not osp.isabs(current_split_file):
            current_split_file = osp.join(cfg.work_dir, current_split_file)
    cfg.query_cfg.current_split_file = current_split_file

    if not osp.isabs(output_split_file):
        output_split_file = osp.join(cfg.work_dir, output_split_file)
    cfg.query_cfg.output_split_file = output_split_file

    cfg.resume = False
    cfg.load_from = None
    if checkpoint_file is not None:
        cfg.load_from = checkpoint_file
    else:
        cfg.query_cfg.type = "Random"

    runner = RUNNERS.build(cfg)

    runner.query()
