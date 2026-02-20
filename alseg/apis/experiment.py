import os.path as osp
import copy

from mmengine import copytree
from mmengine.config import Config
from mmseg.registry import RUNNERS

from .query import query_samples
from .train import train_model
from ..utils import SPLIT_FILE_TXT
from ..utils import Browser, StrategyFigure


def run_experiment(
        cfg: Config,
        init_query_dir: str = None,
        init_train_dir: str = None,
        options: dict = None,
):
    cfg = copy.deepcopy(cfg)
    if options is not None:
        cfg.merge_from_dict(options)

    if cfg.get("experiment_cfg", None) is None:
        raise RuntimeError("'experiment_cfg' could not found the config file.")

    if cfg.query_cfg.type != "Random":
        if init_query_dir is None and init_train_dir is None:
            baseline_work_dir = osp.join(osp.split(osp.split(cfg.work_dir)[0])[0], "Random")
            baseline_browser = Browser(baseline_work_dir, cfg.randomness.seed)
            if osp.exists(baseline_browser.work_dir):
                init_train_dir = baseline_browser.get_train_browser_list()[0].work_dir
                init_query_dir = baseline_browser.get_query_browser_list()[0].work_dir

    cfg.experiment_cfg.num_samples_list = cfg.experiment_cfg.get("num_samples_list", [])
    if len(cfg.experiment_cfg.num_samples_list) == 0:
        cfg.experiment_cfg.num_samples_list.append(cfg.experiment_cfg.init_samples)
        for _ in range(cfg.experiment_cfg.num_cycles):
            cfg.experiment_cfg.num_samples_list.append(cfg.experiment_cfg.num_query)

    main_runner = RUNNERS.build(cfg)
    del cfg.visualizer.name

    checkpoint_file = None
    for cycle_id, num_samples in enumerate(cfg.experiment_cfg.num_samples_list):

        query_name = f"query_{cycle_id}"
        train_name = f"train_{cycle_id}"

        query_dir = osp.join(main_runner.log_dir, query_name)
        train_dir = osp.join(main_runner.log_dir, train_name)

        if cycle_id == 0 and init_query_dir is not None and init_train_dir is not None:
            copytree(init_query_dir, query_dir)
            copytree(init_train_dir, train_dir)
            checkpoint_file = Browser(train_dir).get_best_checkpoint_file()
            continue

        prev_split = None
        if cycle_id > 0:
            prev_split = osp.join(main_runner.log_dir, f"query_{cycle_id - 1}", SPLIT_FILE_TXT)

        next_split = osp.join(query_dir, SPLIT_FILE_TXT)

        main_runner.logger.info(f"Query {cycle_id} started.")

        query_options = dict(
            experiment_name=query_name,
            work_dir=query_dir,
        )

        query_samples(
            cfg,
            num_samples,
            output_split_file=next_split,
            current_split_file=prev_split,
            checkpoint_file=checkpoint_file,
            options=query_options,
        )

        main_runner.logger.info(f"Train {cycle_id} started.")

        train_options = dict(
            experiment_name=train_name,
            work_dir=train_dir,
        )

        train_model(
            cfg,
            split_file=next_split,
            options=train_options,
        )

        checkpoint_file = Browser(train_dir).get_best_checkpoint_file()

    main_runner.logger.info("Experiment finished.")

    # Save experiment results as a figure
    fig = StrategyFigure()
    br = Browser(osp.split(cfg.work_dir)[0])
    fig.add_strategy(br, seed_list=[cfg.randomness.seed])
    fig.plot_curves(save_path=f"{main_runner.log_dir}.png")
