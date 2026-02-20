import os.path as osp

from mmengine.config import Config


def parse_configs(
        config_file: str,
        work_dir: str = None,
        experiment_name: str = None,
        seed: int = None,
        use_single_thread: bool = False,
        options: dict = None,
):
    cfg = Config.fromfile(config_file)
    if options is not None:
        cfg.merge_from_dict(options)

    if cfg.get("experiment_name", None) is None:
        cfg.experiment_name = cfg.query_cfg.type

    if experiment_name is not None:
        cfg.experiment_name = experiment_name.strip()

    cfg.work_dir = osp.join("./logs", osp.splitext(osp.basename(config_file))[0])
    if work_dir is not None:
        cfg.work_dir = work_dir

    cfg.work_dir = osp.join(cfg.work_dir, cfg.experiment_name)

    cfg.randomness = {}
    if seed is not None:
        cfg.randomness = dict(seed=seed, deterministic=True)
        cfg.work_dir = osp.join(cfg.work_dir, f"seed_{seed}")

    if not osp.isabs(cfg.work_dir):
        cfg.work_dir = osp.abspath(cfg.work_dir)

    cfg.launcher = "none"
    cfg.load_from = None
    cfg.resume = False

    if use_single_thread:
        cfg.env_cfg.mp_cfg = {}
        if cfg.get("train_dataloader", None) is not None:
            cfg.train_dataloader.num_workers = 0
            cfg.train_dataloader.persistent_workers = False
        if cfg.get("val_dataloader", None) is not None:
            cfg.val_dataloader.num_workers = 0
            cfg.val_dataloader.persistent_workers = False
        if cfg.get("test_dataloader", None) is not None:
            cfg.test_dataloader.num_workers = 0
            cfg.test_dataloader.persistent_workers = False
        if cfg.get("query_dataloader", None) is not None:
            cfg.query_dataloader.num_workers = 0
            cfg.query_dataloader.persistent_workers = False

    cfg.runner_type = "ActiveLearningRunner"

    return cfg
