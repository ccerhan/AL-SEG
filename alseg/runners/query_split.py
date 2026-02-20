from typing import List

from mmseg.datasets import BaseSegDataset

from .utils import load_img_names, get_img_idxs, save_img_idx


class QuerySplit(object):
    def __init__(self, dataset: BaseSegDataset, split_file: str = None):
        self.dataset = dataset
        self.split_file = split_file
        self.labelled_names = []
        if self.split_file is not None:
            self.labelled_names = load_img_names(dataset, split_file)
        self.all_names = load_img_names(dataset)
        self.labelled_idx, self.unlabelled_idx = get_img_idxs(self.labelled_names, self.all_names)

    def __len__(self):
        return len(self.all_names)

    def save(self, selected_idx: List[int], split_file: str):
        self.labelled_idx.extend(selected_idx)
        labelled_idx = sorted(self.labelled_idx)
        save_img_idx(labelled_idx, self.all_names, split_file)
