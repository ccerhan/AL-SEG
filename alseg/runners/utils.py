import os.path as osp
from typing import List, Tuple

import mmengine
from mmseg.datasets import BaseSegDataset


def load_img_names(dataset: BaseSegDataset, split_file: str = None) -> List[str]:
    img_name_list = []
    if split_file is None:
        img_dir = dataset.data_prefix.get("img_path", None)
        for data in dataset.load_data_list():
            img_name = osp.relpath(data["img_path"], img_dir).replace(dataset.img_suffix, "")
            img_name_list.append(img_name)
    else:
        lines = mmengine.list_from_file(split_file, backend_args=dataset.backend_args)
        for line in lines:
            img_name = line.strip()
            img_name_list.append(img_name)

    img_name_list = sorted(img_name_list)
    return img_name_list


def get_img_idxs(labelled_names: List[str], all_names: List[str]) -> Tuple[List[int], List[int]]:
    labelled_idx, unlabelled_idx = [], []
    for idx, name in enumerate(all_names):
        if name in labelled_names:
            labelled_idx.append(idx)
        else:
            unlabelled_idx.append(idx)
    return labelled_idx, unlabelled_idx


def save_img_idx(img_idx: List[int], all_names: List[str], split_file: str):
    lines = []
    for i, name in enumerate(all_names):
        if i in img_idx:
            lines.append(f"{name}\n")

    with open(split_file, "w") as f:
        f.writelines(lines)
