import argparse

from mmengine.config import DictAction

from alseg.apis import parse_configs, run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run an active learning experiment")

    parser.add_argument("config", type=str,
                        help="config file path")
    parser.add_argument("--work-dir", type=str, required=False,
                        help="the directory to save logs and models")
    parser.add_argument("--experiment-name", type=str, required=False,
                        help="the name of the current experiment")
    parser.add_argument("--seed", type=int, required=False,
                        help="random seed")
    parser.add_argument("--use-single-thread", action='store_true',
                        help="sets number of data loader workers and opencv omp threads to zero")
    parser.add_argument("--options", nargs="+", action=DictAction, required=False,
                        help="override some settings in the used config, the key-value pair "
                             "in xxx=yyy format will be merged into config file. If the value to "
                             "be overwritten is a list, it should be like key='[a,b]' or key=a,b "
                             "It also allows nested list/tuple values, e.g. key='[(a,b),(c,d)]' "
                             "Note that the quotation marks are necessary and that no white space "
                             "is allowed.")

    parser.add_argument("--init-query-dir", type=str, required=False,
                        help="initial query directory")
    parser.add_argument("--init-train-dir", type=str, required=False,
                        help="initial train directory")

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = parse_configs(
        args.config,
        args.work_dir,
        args.experiment_name,
        args.seed,
        args.use_single_thread,
        args.options,
    )

    run_experiment(
        cfg,
        args.init_query_dir,
        args.init_train_dir,
    )


if __name__ == "__main__":
    main()
