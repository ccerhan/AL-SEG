import re
import copy
import json
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from . import Browser


sns.set_style("ticks")


class StrategyFigure(object):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.data = []
        self.metric = "mIoU"

    def add_strategy(self, browser: Browser, seed_list: List[int] = None,
                     display_name: str = None):
        experiment_name = browser.get_experiment_name()
        if display_name is not None:
            experiment_name = display_name

        data = dict(x=[], y=[], label=experiment_name)

        if seed_list is None or len(seed_list) == 0:
            seed_list = browser.get_seed_list()

        y_list = []
        for _, seed in enumerate(seed_list):
            br = Browser(browser.work_dir, seed)
            if not br.valid():
                continue

            data["x"] = [b.get_split_size() for b in br.get_query_browser_list()]

            scores = {}
            for b in br.get_train_browser_list():
                max_val_log = b.get_best_val_scores(self.metric)

                if "all" not in scores:
                    scores["all"] = []
                scores["all"].append(max_val_log[self.metric])

                per_class_metric = self.metric.replace("m", "")
                for c, m in max_val_log["per_class"].items():
                    if c not in scores:
                        scores[c] = []
                    scores[c].append(m[per_class_metric])

            y_list.append(scores)

        if len(y_list) > 0:
            data["y"] = y_list

            self.data.append(data)

        return copy.deepcopy(data)

    def add_full_training(self, browser: Browser, seed_list: List[int] = None):
        data = dict(x=[], y=[], label="Full")

        if seed_list is None or len(seed_list) == 0:
            seed_list = browser.get_seed_list()

        y_list = []
        for _, seed in enumerate(seed_list):
            br = Browser(browser.work_dir, seed)
            if not br.valid():
                continue

            max_val_log = br.get_best_val_scores(self.metric)

            scores = dict(all=[])
            scores["all"].append(max_val_log[self.metric])

            per_class_metric = self.metric.replace("m", "")
            for c, m in max_val_log["per_class"].items():
                if c not in scores:
                    scores[c] = []
                scores[c].append(m[per_class_metric])

            y_list.append(scores)

        if len(y_list) > 0:
            data["x"] = [0]
            data["y"] = y_list

            self.data.append(data)

        return copy.deepcopy(data)

    def save_results(self, file_path: str, discard_class_results: bool = False):
        if ".json" not in file_path:
            raise ValueError("File extension must be '.json'")

        data = copy.deepcopy(self.data)
        if discard_class_results:
            for i, d in enumerate(self.data):
                for j, dy in enumerate(d["y"]):
                    for key, _ in dy.items():
                        if key != "all":
                            del data[i]["y"][j][key]

        with open(file_path, "w") as fp:
            s = json.dumps(data, indent=4)
            s = re.sub(r'\n +([0-9-\]])', r' \1', s)
            fp.write(s)

    def load_results(self, file_path: str):
        if ".json" not in file_path:
            raise ValueError("File extension must be '.json'")

        with open(file_path, "r") as fp:
            self.data = json.load(fp)

    def plot_curves_ax(self, ax, **kwargs):
        if len(self.data) == 0 or (len(self.data) == 1 and self.data[0]["label"] == "Full"):
            raise ValueError("At least one 'strategy' should be added.")

        cls = kwargs.get("cls", "all")
        title = kwargs.get("title", None)
        dataset_size = kwargs.get("dataset_size", None)
        cycles = kwargs.get("cycles", (0, 6))
        show_std = kwargs.get("show_std", True)
        show_percent = kwargs.get("show_percent", 95)
        show_initial_score = kwargs.get("show_initial_score", False)
        ylim = kwargs.get("ylim", None)

        font_size = 11

        base_score_avg, base_score_std = None, None

        min_y, max_y = 9999, -9999
        for i, data in enumerate(self.data):
            x, y = data["x"], data["y"]
            label = data["label"]
            x = np.array(x)

            scores = np.array([s[cls] for s in y])
            avg = np.mean(scores, axis=0)
            std = np.std(scores, axis=0)

            if label == "Full":
                avg[0] = avg[0] * (show_percent / 100)

            min_cycle, max_cycle = cycles
            if min_cycle is None:
                min_cycle = 0
            if max_cycle is None:
                max_cycle = len(x)

            if label != "Full":
                base_score_avg, base_score_std = float(avg[0]), float(std[0])
                x = x[min_cycle:max_cycle + 1]
                avg = avg[min_cycle:max_cycle + 1]
                std = std[min_cycle:max_cycle + 1]

            if np.min(avg - std) < min_y:
                min_y = np.min(avg - std)
            if np.max(avg + std) > max_y:
                max_y = np.max(avg + std)

            if i == len(self.data) - 1:
                xlabels = []
                for n in x:
                    xlabel = str(n)
                    if dataset_size is not None:
                        xlabel = f"{xlabel}\n({round((n * 100) / dataset_size)}%)"
                    xlabels.append(xlabel)
                ax.set_xticks(x)
                ax.set_xticklabels(xlabels)

            if label == "Full":
                x = np.array(self.data[i + 1]["x"])[min_cycle:max_cycle + 1]
                full_label = f"{label}-mIoU ({show_percent}%)"
                ax.plot([x[0], x[-1]], [avg[0], avg[0]], "k--", alpha=0.45, label=full_label)
            else:
                ax.plot(x, avg, "o-", markersize=4, label=label)
                if show_std:
                    # ax.errorbar(x, avg, yerr=std)
                    ax.fill_between(x, avg - std, avg + std, alpha=0.1, linewidth=0, antialiased=True)

        if dataset_size is not None:
            ax.set_xlabel("# of samples (% of the data)", fontsize=font_size)
        else:
            ax.set_xlabel("# of samples", fontsize=font_size)
        ax.set_ylabel(f"{self.metric} (%)", fontsize=font_size)

        min_y = min_y - (max_y - min_y) * 0.01
        max_y = max_y + (max_y - min_y) * 0.01

        if ylim is None:
            ax.set_ylim((min_y, max_y))
        else:
            ax.set_ylim(ylim)

        ax.legend(loc="lower right")

        plot_title = ""
        if title is not None:
            plot_title = title

        if cls != "all":
            plot_title = f"{plot_title} | class: {cls}".strip()

        if show_initial_score:
            base_score = fr"{round(base_score_avg, 2)} $\pm$ {round(base_score_std, 2)}"
            plot_title = f"{plot_title} (initial {self.metric}: {base_score})".strip()

        ax.set_title(plot_title, fontsize=font_size + 1)

    @staticmethod
    def compute_ppm(scores: np.ndarray, p_value: float = 0.1) -> Tuple[np.ndarray, float]:
        """
        scores: n_methods, n_runs, n_cycles
        """

        def calculate_t_value(i_scores: np.ndarray, j_scores: np.ndarray):
            T = len(i_scores)
            m = np.mean(i_scores - j_scores)
            s = np.sqrt(np.sum((i_scores - j_scores - m) ** 2) / (T - 1))
            t = (np.sqrt(T) * m) / (s + np.finfo(float).eps)
            return t

        n_methods, n_runs, n_cycles = scores.shape
        t_interval = stats.t.ppf(1 - p_value / 2, n_runs - 1)

        ppm = np.zeros((n_methods + 1, n_methods))
        for cyc in range(n_cycles):
            for i, si in enumerate(scores):
                for j, sj in enumerate(scores):
                    if i >= j:
                        continue
                    t_cyc = calculate_t_value(si[:, cyc], sj[:, cyc])
                    if t_cyc >= t_interval:
                        ppm[i, j] += (1 / n_cycles)
                    if t_cyc <= -t_interval:
                        ppm[j, i] += (1 / n_cycles)

        ppm[-1, :] = np.mean(ppm[:-1, :], axis=0)  # create row-wise average (phi)
        return ppm, t_interval

    def create_ppm(self, **kwargs) -> Tuple[np.ndarray, List[str]]:
        if len(self.data) < 2:
            raise ValueError("At least two 'strategy' should be added.")

        cls = kwargs.get("cls", "all")
        cycles = kwargs.get("cycles", (1, 6))
        p_value = kwargs.get("p_value", 0.05)  # (1 - p_value) confidence level

        method_names, method_scores = [], []
        for _, data in enumerate(self.data):
            label = data["label"]
            if label == "Full":
                continue
            method_names.append(label)
            method_scores.append(np.array([s[cls] for s in data["y"]]))

        num_seeds = [v.shape[0] for v in method_scores]
        for s1, s2 in zip(num_seeds[:-1], num_seeds[1:]):
            if s1 != s2:
                raise ValueError("Number of seeds is not equal")

        scores = np.array([s[:, cycles[0]:cycles[1]] for s in method_scores])
        ppm, _ = self.compute_ppm(scores, p_value)

        return ppm, method_names

    @staticmethod
    def plot_ppm_axis(ax, ppm: np.ndarray, methods: List[str], ppm_aspect: float = 0.45):
        if ppm_aspect < 0.5:
            xticks_rotation = 0
            ticks_font_size = 12
            values_font_size = 10
            color_bar_shrink = 0.5
        else:
            xticks_rotation = 90
            ticks_font_size = 10
            values_font_size = 8
            color_bar_shrink = 0.75

        im = ax.matshow(ppm, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(methods)), [f"{m.split(' ')[0]}" for m in methods], fontsize=ticks_font_size)
        ax.set_yticks(np.arange(len(methods) + 1), [f"{m}" for m in methods] + ["$\\Phi$"], fontsize=ticks_font_size)
        ax.set_aspect(ppm_aspect)
        ax.spines[:].set_visible(False)
        ax.set_yticks([ppm.shape[0] - 1.5], minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=5)
        ax.xaxis.set_tick_params(rotation=xticks_rotation, bottom=False, top=False)
        ax.yaxis.set_tick_params(left=False, right=False)
        ax.yaxis.set_tick_params(which="minor", left=False, right=False)

        for i in range(len(methods) + 1):
            for j in range(len(methods)):
                color = "k" if ppm[i, j] > 0.5 else "w"
                ax.text(j, i, f"{ppm[i, j]:.2f}", ha="center", va="center", color=color, fontsize=values_font_size)

        cbar = ax.figure.colorbar(im, ax=ax, aspect=35, shrink=color_bar_shrink)
        cbar.ax.tick_params(labelsize=9)

    def plot_ppm_ax(self, ax, **kwargs):
        ppm_aspect = kwargs.get("ppm_aspect", 0.45)

        ppm, methods = self.create_ppm(**kwargs)
        self.plot_ppm_axis(ax, ppm, methods, ppm_aspect)

    def plot_curves(self, **kwargs):
        save_path = kwargs.get("save_path", None)

        fig, ax = plt.subplots(**self.params)
        self.plot_curves_ax(ax, **kwargs)

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

    def plot_ppm(self, **kwargs):
        save_path = kwargs.get("save_path", None)

        fig, ax = plt.subplots(**self.params)
        self.plot_ppm_ax(ax, **kwargs)

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
