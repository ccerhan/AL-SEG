import os.path as osp
import json
from typing import Union, List

from mmengine.config import Config
from mmengine import list_dir_or_file

from .constants import SPLIT_FILE_TXT, VALUES_FILE_NPY


class Browser(object):
    def __init__(self, work_dir: str, seed: int = None):
        self.work_dir = work_dir
        self.seed = seed
        if seed is not None and f"seed_{seed}" not in work_dir:
            self.work_dir = osp.join(work_dir, f"seed_{seed}")
        if "seed_" in work_dir:
            self.seed = int(work_dir.split("seed_")[1].split("/")[0])

    def valid(self):
        return osp.exists(self.work_dir) and osp.isdir(self.work_dir)

    def get_experiment_name(self):
        if "seed_" in self.work_dir:
            return osp.split(self.work_dir.split("/seed_")[0])[1]
        else:
            return osp.split(self.work_dir)[1]

    def get_seed_list(self):
        seed_dirs = list_dir_or_file(self.work_dir, list_dir=True, list_file=False)
        seed_vals = [int(s.replace("seed_", "")) for s in seed_dirs if s.startswith("seed")]
        return sorted(seed_vals)

    def get_timestamp_list(self):
        ts = list_dir_or_file(self.work_dir, list_dir=True, list_file=False)
        return sorted(ts, reverse=True)

    def get_latest_timestamp(self):
        ts = self.get_timestamp_list()
        return ts[0]

    def get_cfg(self) -> Union[Config, None]:
        files = list_dir_or_file(self.work_dir, list_dir=False, list_file=True, suffix=".py")
        paths = [osp.join(self.work_dir, fn) for fn in files]
        return Config.fromfile(paths[0]) if len(paths) > 0 else None

    def get_log_lines(self, timestamp: str = None) -> List[str]:
        timestamp = timestamp if timestamp is not None else self.get_latest_timestamp()
        path = osp.join(self.work_dir, timestamp, f"{timestamp}.log")
        with open(path) as fp:
            return [n.strip() for n in fp.readlines()]

    def get_sub_browser_list(self, prefix: str = "", timestamp: str = None) -> List["Browser"]:
        timestamp = timestamp if timestamp is not None else self.get_latest_timestamp()
        path = osp.join(self.work_dir, timestamp)
        if not osp.exists(path):
            return []
        dirs = list_dir_or_file(path, list_dir=True, list_file=False)
        dirs = [d for d in dirs if d.startswith(prefix)]
        return [Browser(osp.join(path, d)) for d in sorted(dirs)]

    def get_query_browser_list(self, timestamp: str = None) -> List["Browser"]:
        return self.get_sub_browser_list("query", timestamp)

    def get_train_browser_list(self, timestamp: str = None) -> List["Browser"]:
        return self.get_sub_browser_list("train", timestamp)

    def get_best_checkpoint_file(self) -> Union[str, None]:
        files = list_dir_or_file(self.work_dir, list_dir=False, list_file=True)
        paths = [osp.join(self.work_dir, fn) for fn in files if fn.startswith("best")]
        return paths[0] if len(paths) > 0 else None

    def get_split_file(self) -> Union[str, None]:
        files = list_dir_or_file(self.work_dir, list_dir=False, list_file=True, suffix=SPLIT_FILE_TXT)
        paths = [osp.join(self.work_dir, fn) for fn in files]
        return paths[0] if len(paths) > 0 else None

    def get_split_size(self) -> Union[int, None]:
        split_file = self.get_split_file()
        if split_file is not None:
            with open(split_file) as fp:
                return len(fp.readlines())
        return None

    def get_values_file(self, timestamp: str = None) -> Union[str, None]:
        timestamp = timestamp if timestamp is not None else self.get_latest_timestamp()
        values_dir = osp.join(self.work_dir, timestamp)
        files = list_dir_or_file(values_dir, list_dir=False, list_file=True, suffix=VALUES_FILE_NPY)
        paths = [osp.join(values_dir, fn) for fn in files]
        return paths[0] if len(paths) > 0 else None

    def get_file_path(self, file_name: str):
        return osp.join(self.work_dir, file_name)

    def get_epoch_logs(self, timestamp: str = None) -> Union[List[dict], None]:
        timestamp = timestamp if timestamp is not None else self.get_latest_timestamp()
        path = osp.join(self.work_dir, timestamp, "vis_data", f"{timestamp}.json")
        if not osp.exists(path):
            return None
        with open(path) as f:
            logs = [json.loads(line) for line in f.readlines()]
        for log in logs:
            if "epoch" in log:
                log["mode"] = "train"
            else:
                log["mode"] = "val"
                log["epoch"] = log["step"]
        return logs

    def get_best_val_scores(self, metric: str = "mIoU", timestamp: str = None) -> Union[dict, None]:
        logs = self.get_epoch_logs(timestamp)
        max_val_log = None
        if logs is not None:
            val_logs = [log for log in logs if log["mode"] == "val"]
            max_val_log = max(val_logs, key=lambda log: log[metric])
            lines = self.get_log_lines(timestamp)

            # per class scores
            max_val_epoch = max_val_log["epoch"]
            epoch_found = False
            for i, line in enumerate(lines):
                if f"Saving checkpoint at {max_val_epoch} epochs" in line:
                    epoch_found = True
                if f"Epoch(val) [{max_val_epoch}]" in line:
                    epoch_found = True
                if epoch_found and "per class results:" in line:
                    headers = [v.strip() for v in lines[i + 3].split("|") if v.strip() != ""]
                    per_class_scores = {}
                    for row in lines[i + 5:]:
                        if row.startswith("+"):
                            break
                        metric = [v.strip() for v in row.split("|") if v.strip() != ""]
                        class_name = metric[0]
                        per_class_scores[class_name] = dict([(h, float(m)) for h, m in zip(headers[1:], metric[1:])])
                    max_val_log["per_class"] = per_class_scores
                    break

        return max_val_log

    def get_selected_idx_list(self, timestamp: str = None) -> Union[List[int], None]:
        lines = self.get_log_lines(timestamp)
        idx_list = None
        for i, line in enumerate(lines):
            if "Selected set indexes" in line:
                idx_list = json.loads(lines[i + 1])
        return idx_list
