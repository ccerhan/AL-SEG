import argparse

from mmengine.config import DictAction

from alseg.apis import parse_configs, query_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Query samples from a dataset")

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

    parser.add_argument("--num-samples", type=int, required=True,
                        help="number of samples to query")
    parser.add_argument("--output-split", type=str, required=True,
                        help="output train split file path (.txt)")
    parser.add_argument("--current-split", type=str, required=False,
                        help="current train split file path (.txt)")
    parser.add_argument("--checkpoint", type=str, required=False,
                        help="model checkpoint file path (.pth)")

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

    query_samples(
        cfg,
        args.num_samples,
        args.output_split,
        args.current_split,
        args.checkpoint,
    )


if __name__ == "__main__":
    main()
