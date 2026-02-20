import pickle
from typing import List, Tuple

import numpy as np
import torch


class ConformalRiskControl(object):
    def __init__(
            self,
            num_cls: int,
            ignore_index: int = None,
            risk_resolution: int = 100,
            device: str = "cpu",
    ):
        self.num_cls = num_cls
        self.ignore_index = ignore_index
        self.risk_resolution = risk_resolution
        self.device = device
        self.conf_mat_list = []
        self.conf_mat_table = None
        self.num_data = None

    @staticmethod
    def load(path: str, device: str = None) -> "ConformalRiskControl":
        with open(path, "rb") as fp:
            obj = pickle.load(fp)
        if device is not None:
            obj.device = device
        obj.conf_mat_table = obj.conf_mat_table.to(obj.device)
        obj.num_data = obj.conf_mat_table.shape[0]
        return obj

    def save(self, path: str) -> None:
        if self.conf_mat_table is not None:
            self.conf_mat_table = self.conf_mat_table.detach().cpu()
            with open(path, "wb") as fp:
                pickle.dump(self, fp)
            self.conf_mat_table = self.conf_mat_table.to(self.device)

    def append(self, values: torch.Tensor, labels: torch.Tensor) -> None:
        ignore_mask = (labels == self.ignore_index) if self.ignore_index is not None else None

        conf_mat = torch.zeros(
            (self.num_cls, self.risk_resolution + 1, 4),
            device=self.device,
            dtype=torch.int64
        )

        for cls_idx in range(self.num_cls):
            for j, lam in enumerate(range(self.risk_resolution + 1)):
                lam = lam / self.risk_resolution
                pred_masks = values[:, cls_idx]
                pred_masks = (pred_masks >= lam)

                if ignore_mask is not None:
                    pred_masks[ignore_mask] = False

                true_masks = (labels == cls_idx)
                tp_fn_fp_tn = self.confusion_matrix(pred_masks, true_masks)
                conf_mat[cls_idx, j, :] = tp_fn_fp_tn[0]

        self.conf_mat_list.append(conf_mat)

    def done(self):
        self.conf_mat_table = torch.stack(self.conf_mat_list)
        self.num_data = len(self.conf_mat_list)
        self.conf_mat_list.clear()

    @staticmethod
    def confusion_matrix(pred_masks: torch.Tensor, true_masks: torch.Tensor) -> torch.Tensor:
        n, w, h = true_masks.size()
        total = w * h
        p = true_masks.sum(dim=(1, 2))
        pp = pred_masks.sum(dim=(1, 2))
        tp = (pred_masks * true_masks).sum(dim=(1, 2))
        fn = p - tp
        fp = pp - tp
        tn = total - p - fp
        return torch.cat([tp, fn, fp, tn], dim=0).view(-1, n).T

    def calculate_risk(self, cls_idx: int, risk_func: str = "fnr") -> Tuple[int, List[float], List[float]]:
        tp = self.conf_mat_table[:, cls_idx, :, 0]
        fn = self.conf_mat_table[:, cls_idx, :, 1]
        fp = self.conf_mat_table[:, cls_idx, :, 2]
        tn = self.conf_mat_table[:, cls_idx, :, 3]

        if risk_func.lower() == "fnr":
            # false negative rate (miss rate): false negatives over all positives
            risk = fn / (tp + fn)
            # exclude the samples in which there are no positives for this class
            risk = risk[torch.bitwise_not(torch.isnan(risk))].view(-1, self.risk_resolution + 1)

        elif risk_func.lower() == "fpr":
            # false positive rate (fall-out): false positives over all negatives
            risk = fp / (tn + fp)
            # exclude the samples in which there are no negatives for this class
            risk = risk[torch.bitwise_not(torch.isnan(risk))].view(-1, self.risk_resolution + 1)

        elif risk_func.lower() == "f1":  # non-monotonic function - not working properly
            risk = (2 * tp) / (2 * tp + fp + fn + torch.finfo(torch.float32).eps)

        elif risk_func.lower() == "iou":  # non-monotonic function - not working properly
            risk = tp / (tp + fp + fn + torch.finfo(torch.float32).eps)

        else:
            raise ValueError(f"Invalid risk function: {risk_func}")

        num = len(risk)
        if num == 0:
            return 0, None, None

        risk_std, risk_avg = torch.std_mean(risk, dim=0)
        risk_avg = risk_avg.detach().cpu().tolist()
        risk_std = risk_std.detach().cpu().tolist()

        return num, risk_avg, risk_std

    def calibrate(self, alpha: float, risk_func: str = "fnr", precision: int = 6) -> Tuple[List[int], List[float]]:

        def find_lamhat(a: float, n: int, r: List[float], prec: int = 6):
            if n == 0:
                return 0

            c = (n + 1) / n * a - 1 / (n + 1)
            if c <= 0:
                return 0

            yp = np.array(r) - c
            xp = np.linspace(0, 1, self.risk_resolution + 1)
            lam = np.linspace(0, 1, 10 ** prec + 1)
            risk = np.interp(lam, xp, yp)

            if np.min(risk) > 0:
                raise RuntimeError(f"Could not found lambda hat: {risk_func.upper()} (alpha: {a})")

            lamhat = float(lam[np.argmin(np.abs(risk))])
            lamhat = round(lamhat, prec)

            # import matplotlib.pyplot as plt
            # plt.plot(1 - lam, risk + c, "C0-", label=f"{risk_func.upper()}")
            # plt.plot(1 - xp, yp + c, "C1.")
            # plt.plot([0, 1], [c, c], "C5-.", alpha=0.5, label=r"$\alpha - \frac{1 - \alpha}{n}$")
            # plt.plot([1 - lamhat, 1 - lamhat], [0, 1], "C3-", alpha=0.5, label=r"$\hat{\lambda}$")
            # plt.xlabel(r"$\lambda$")
            # plt.ylabel(f"Empirical Risk")
            # plt.tight_layout()
            # plt.legend()
            # plt.grid()
            # plt.show()

            return lamhat

        nums = []
        lamhat = []
        for cls_idx in range(self.num_cls):
            n, risk_avg, risk_std = self.calculate_risk(cls_idx, risk_func)
            nums.append(n)

            lam = find_lamhat(alpha, n, risk_avg, precision)
            lamhat.append(lam)

        return nums, lamhat

    @staticmethod
    def apply(lamhat: List[float], values: torch.Tensor, ignore_mask: torch.Tensor = None) -> torch.Tensor:
        pred_sets = []
        for cls_idx, lam in enumerate(lamhat):
            pred_masks = values[:, cls_idx]
            pred_masks = (pred_masks >= lam)

            if ignore_mask is not None:
                pred_masks[ignore_mask] = False

            pred_sets.append(pred_masks)
        pred_sets = torch.stack(pred_sets)

        return torch.transpose(pred_sets, 0, 1)

    @staticmethod
    def lengths(pred_sets: torch.Tensor, ignore_mask: torch.Tensor = None) -> torch.Tensor:
        b, c, h, w = pred_sets.size()
        pred_sets = pred_sets.view(b, c, -1)

        valid_mask = ~ignore_mask.view(b, -1) if ignore_mask is not None else None

        lengths = torch.zeros((b, c + 1), dtype=torch.int64, device=pred_sets.device)
        for i in range(b):
            if valid_mask is not None:
                pred_vals = pred_sets[i, :, valid_mask[i]]
            else:
                pred_vals = pred_sets[i]

            set_lengths = pred_vals.sum(dim=0)
            length, cnt = torch.unique(set_lengths, return_counts=True)
            for len, num in zip(length, cnt):
                lengths[i, len] = num

        return lengths

    @staticmethod
    def coverage(pred_sets: torch.Tensor, labels: torch.Tensor, ignore_mask: torch.Tensor = None) -> torch.Tensor:
        b, c, h, w = pred_sets.size()
        pred_sets = pred_sets.view(b, c, -1)
        labels = labels.view(b, -1)

        valid_mask = ~ignore_mask.view(b, -1) if ignore_mask is not None else None

        coverage_list = []
        for i in range(b):
            if valid_mask is not None:
                pred_vals = pred_sets[i, :, valid_mask[i]]
                true_vals = labels[i, valid_mask[i]]
            else:
                pred_vals = pred_sets[i]
                true_vals = labels[i]

            true_vals = true_vals.unsqueeze(dim=0)
            is_covered = torch.gather(pred_vals, 0, true_vals)
            coverage = is_covered.sum() / is_covered.shape[1]
            coverage_list.append(coverage)

        return torch.vstack(coverage_list)


class ConformalRiskEvaluator(ConformalRiskControl):
    def __init__(
            self,
            num_cls: int,
            lamhat: List[float],
            ignore_index: int = None,
            device: str = "cpu",
    ):
        super().__init__(num_cls, ignore_index, risk_resolution=0, device=device)
        self.lamhat = lamhat
        self.coverage_list = []
        self.length_list = []
        self.coverage_table = None
        self.length_table = None

    def append(self, values: torch.Tensor, labels: torch.Tensor) -> None:
        ignore_mask = (labels == self.ignore_index) if self.ignore_index is not None else None

        conf_mat = torch.zeros(
            (self.num_cls, self.risk_resolution + 1, 4),
            device=self.device,
            dtype=torch.int64
        )

        pred_sets = []
        for cls_idx in range(self.num_cls):
            lam = self.lamhat[cls_idx]
            pred_masks = values[:, cls_idx]
            pred_masks = (pred_masks >= lam)
            pred_sets.append(pred_masks)

            if ignore_mask is not None:
                pred_masks[ignore_mask] = False

            true_masks = (labels == cls_idx)
            tp_fn_fp_tn = self.confusion_matrix(pred_masks, true_masks)
            conf_mat[cls_idx, 0, :] = tp_fn_fp_tn[0]

        self.conf_mat_list.append(conf_mat)

        pred_sets = torch.stack(pred_sets)
        pred_sets = torch.transpose(pred_sets, 0, 1)

        coverage = ConformalRiskControl.coverage(pred_sets, labels, ignore_mask)
        self.coverage_list.append(coverage[0])

        lengths = ConformalRiskControl.lengths(pred_sets, ignore_mask)
        self.length_list.append(lengths[0])

    def done(self):
        super().done()
        self.coverage_table = torch.stack(self.coverage_list)
        self.coverage_list.clear()
        self.length_table = torch.stack(self.length_list)
        self.length_list.clear()

    def evaluate(self, risk_func: str = "fnr", precision: int = 6) -> Tuple[List[int], List[float], List[float]]:
        nums, risks, stds = [], [], []
        for cls_idx in range(self.num_cls):
            n, risk_avg, risk_std = self.calculate_risk(cls_idx, risk_func)

            nums.append(n)
            risks.append(None)
            stds.append(None)

            if n > 0:
                risks[cls_idx] = round(risk_avg[0], precision)

            if n > 1:
                stds[cls_idx] = round(risk_std[0], precision)

        return nums, risks, stds

    def mean_coverage(self) -> float:
        return float(self.coverage_table.mean())

    def mean_length(self) -> float:
        bins = torch.arange(self.num_cls + 1).repeat(self.num_data, 1).to(self.length_table.device)
        length = (self.length_table * bins).sum(dim=1) / self.length_table.sum(dim=1)
        return float(length.mean())

    def mean_length_pdf(self) -> List[float]:
        pdf = self.length_table / self.length_table.sum(dim=1).unsqueeze(dim=1)
        return pdf.mean(dim=0).cpu().tolist()
