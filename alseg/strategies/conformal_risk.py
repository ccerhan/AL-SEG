import random
import os.path as osp
from typing import Dict, List, Union, Tuple

import torch
from torch.utils.data import DataLoader
from prettytable import PrettyTable

from mmengine.registry import LOOPS
from mmengine.device import get_device
from mmseg.datasets import BaseSegDataset

from ..runners import QueryLoop
from ..helpers import ConformalRiskControl, ConformalRiskEvaluator, ConformalRiskMap, UncertaintyMap
from ..selectors import TopDiverseK, KCenterGreedy


@LOOPS.register_module()
class ConformalRisk(QueryLoop):
    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            num_samples: int,
            current_split_file: str = None,
            output_split_file: str = None,
            fp16: bool = False,
            ignore_type: str = "none",
            alpha: float = 0.05,
            risk_function: str = "fnr",
            risk_resolution: int = 1000,
            calib_dataset: str = "train",
            calib_size: int = None,
            selection_type: str = "topdivk",
            eval_risk: bool = False,
            tau: float = 0.5,
    ) -> None:
        super().__init__(runner, dataloader, num_samples, current_split_file, output_split_file, fp16, ignore_type)
        self.alpha = alpha
        self.risk_function = risk_function
        self.risk_resolution = risk_resolution
        self.calib_dataset = calib_dataset
        self.calib_size = calib_size
        self.selection_type = selection_type
        self.eval_risk = eval_risk
        self.tau = tau

    def run(self) -> None:
        calib_idx = None
        if self._DEBUG:
            base_path = self.runner.cfg.load_from.split("train_")
            calib_path = osp.join(base_path[0], f"query_{int(base_path[1][0]) + 1}", "calibrator.pkl")
            calibrator = ConformalRiskControl.load(calib_path, device=get_device())
        else:
            dataset_idx_pairs = []
            if "train" in self.calib_dataset:
                calib_idx = [i for i in self.query_split.labelled_idx]
                if self.calib_size is not None:
                    random.shuffle(calib_idx)
                    calib_idx = calib_idx[:self.calib_size]
                dataset_idx_pairs.append((self.query_split.dataset, calib_idx))
            if "unlabelled" in self.calib_dataset:
                calib_idx = [i for i in self.query_split.unlabelled_idx]
                if self.calib_size is not None:
                    random.shuffle(calib_idx)
                    calib_idx = calib_idx[:self.calib_size]
                dataset_idx_pairs.append((self.query_split.dataset, calib_idx))
            if "val" in self.calib_dataset:
                dataset_idx_pairs.append((self.runner.val_dataloader.dataset, None))
            if len(dataset_idx_pairs) == 0:
                raise ValueError("Invalid calibration dataset")
            calibrator = self.create_calibrator(dataset_idx_pairs, self.risk_resolution)
            calibrator_path = osp.join(self.runner.work_dir, "calibrator.pkl")
            calibrator.save(calibrator_path)

        alpha_list = [self.alpha]
        risk_func_list = [self.risk_function]
        calib_results = self.calibrate(calibrator, alpha_list, risk_func_list)
        self.meta["calibration"] = calib_results

        if not self._DEBUG and self.eval_risk:
            # evaluate on the val set
            if "val" not in self.calib_dataset:
                eval_dataset, eval_idx = self.runner.val_dataloader.dataset, None
                eval_results = self.evaluate(calib_results, eval_dataset, eval_idx)
                self.meta["evaluation"] = eval_results

            # evaluate on the unlabelled set
            if "train" in self.calib_dataset or "val" in self.calib_dataset:
                eval_idx = [i for i in self.query_split.unlabelled_idx]
                eval_results = self.evaluate(calib_results, self.query_split.dataset, eval_idx)
                self.meta["evaluation_unlabelled"] = dict(
                    results=eval_results,
                    calid_idx=calib_idx,
                    eval_idx=eval_idx,
                )

            # evaluate on the remaining labelled set
            if "unlabelled" in self.calib_dataset:
                eval_idx = set([i for i in self.query_split.unlabelled_idx])
                eval_idx = list(eval_idx.difference(set(calib_idx)))
                eval_results = self.evaluate(calib_results, self.query_split.dataset, eval_idx)
                self.meta["evaluation_remaining"] = dict(
                    results=eval_results,
                    calid_idx=calib_idx,
                    eval_idx=eval_idx,
                )

        super().run()

    @torch.no_grad()
    def compute_iter(self, idx: int, data_batch: Dict) -> torch.Tensor:
        seg_results = super().compute_iter(idx, data_batch)
        seg_logits = torch.stack([r.seg_logits.data for r in seg_results])
        probs = torch.softmax(seg_logits, dim=1)

        ignore_mask, valid_mask = None, None
        if self.ignore_type == "index" and self.query_split.dataset.ignore_index is not None:
            labels = torch.vstack([r.gt_sem_seg.data for r in seg_results]).to(get_device())
            ignore_mask = (labels == self.query_split.dataset.ignore_index)
            valid_mask = ~ignore_mask

        return self._compute_iter(idx, probs, ignore_mask, valid_mask, seg_results)

    def _compute_iter(self, idx, probs, ignore_mask, valid_mask, seg_results):
        risk_maps = []
        cooccur_pdfs = []
        conf_weights = []
        for alpha_idx in range(len(self.meta["calibration"]["alpha"])):
            lamhat_fnr = self.meta["calibration"]["fnr"]["data"][alpha_idx]["threshold"]
            lamhat = [max(lam, 0.001) for lam in lamhat_fnr]

            pred_sets = ConformalRiskControl.apply(lamhat, probs, ignore_mask)

            crm = ConformalRiskMap(pred_sets, valid_mask, device=get_device())

            cooccur_pdf = crm.create_cooccur_pdf()
            cooccur_pdfs.append(cooccur_pdf)

            weights = crm.compute_confusion_weights(cooccur_pdf)
            conf_weights.append(weights)

            risk_map = crm.create_risk_map(weights)
            risk_map[torch.isnan(risk_map)] = 0  # if weights.sum() is 0 - means that no confusion, so no risk
            risk_maps.append(risk_map)

        risk_maps = torch.stack(risk_maps)
        risk_map = risk_maps.mean(dim=0)

        um = UncertaintyMap(valid_mask, device=get_device())

        values = um.compute_mean_uncertainty(risk_map)

        cooccur_pdfs = torch.stack(cooccur_pdfs)
        cooccur_pdf = cooccur_pdfs.mean(dim=0)
        embedding = cooccur_pdf.view(cooccur_pdf.shape[0], -1)
        values = torch.hstack([values, embedding])

        self._debug(idx, risk_map, values, seg_results, cooccur_pdf)

        return values

    def select(self, num_samples: int, values: torch.Tensor, unlabelled_idx: torch.LongTensor) -> List[int]:
        if self.selection_type == "topdivk":
            tdk = TopDiverseK(self.selection_type, self.tau)
            selected_idx = tdk.select(num_samples, values, unlabelled_idx)
        elif self.selection_type == "kcenter":
            kc = KCenterGreedy(distance_metric="euclidean")
            kc.weights = values[:, 0]
            kc.tau = self.tau
            embeddings = values[:, 1:]
            selected_idx = kc.select(num_samples, embeddings, unlabelled_idx)
        else:
            raise ValueError("Invalid selection_type")

        return selected_idx

    def _debug(self, idx: int, risk_map, values, seg_results, cooccur_pdf):
        if not self._DEBUG:
            return

        import cv2
        from ..utils.debug import create_debug_image

        cls_names = self.query_split.dataset.METAINFO["classes"]
        palette = self.query_split.dataset.METAINFO["palette"]

        b, _, _ = risk_map.size()
        for i in range(b):
            image_idx = i + idx * b
            is_labelled = image_idx in self.query_split.labelled_idx

            ignore_mask = None
            if self.ignore_type == "index":
                ignore_mask = (seg_results[i].gt_sem_seg.data[0] == self.query_split.dataset.ignore_index).cpu().numpy()

            if values is not None:
                print(f"Idx: {image_idx:4} | Labelled: {int(is_labelled)} | Value: {float(values[i, 0]):.6f}")

            uncert_img = risk_map[i].cpu().numpy()
            cooccur_img = cooccur_pdf[i].cpu().numpy()
            img = create_debug_image(uncert_img, seg_results[i], ignore_mask, cooccur_img, cls_names, palette)

            cv2.imshow("DEBUG", img)
            cv2.waitKey(0)

    def create_calibrator(
            self,
            dataset_idx_pairs: List[Tuple[BaseSegDataset, List[int]]],
            risk_resolution: int = 100,
    ) -> ConformalRiskControl:
        log_interval = self.runner.cfg.default_hooks["logger"]["interval"]
        num_cls = self.runner.model.num_classes
        ignore_index = dataset_idx_pairs[0][0].ignore_index

        calibrator = ConformalRiskControl(num_cls, ignore_index, risk_resolution, device=get_device())

        self.runner.model.eval()
        for dataset_id, (dataset, data_idx) in enumerate(dataset_idx_pairs):
            if data_idx is None:
                data_idx = range(len(dataset))
            calib_size = len(data_idx)

            for i, idx in enumerate(data_idx):
                data_batch = dataset.prepare_data(idx)
                inputs, data_samples = data_batch["inputs"], data_batch["data_samples"]
                data_batch = dict(inputs=[inputs], data_samples=[data_samples])
                seg_results = super().compute_iter(idx, data_batch)
                seg_logits = torch.stack([r.seg_logits.data for r in seg_results])

                probs = torch.softmax(seg_logits, dim=1)
                labels = torch.vstack([r.gt_sem_seg.data for r in seg_results]).to(get_device())

                calibrator.append(probs, labels)

                if i % log_interval == log_interval - 1:
                    i = str(i + 1).rjust(len(str(calib_size)), " ")
                    self.runner.logger.info(f"Epoch(calibration) [{dataset_id}][{i}/{calib_size}]")

        calibrator.done()

        return calibrator

    def calibrate(
            self,
            calibrator: ConformalRiskControl,
            alpha_list: List[float],
            risk_func_list: List[str],
            risk_precision: int = 6,
    ) -> Dict:

        calib_results = dict(alpha=alpha_list)
        for risk_func in risk_func_list:

            calib_risk_results = dict(
                risk=risk_func,
                size=calibrator.num_data,
                name=self.query_split.dataset.METAINFO["classes"],
                data=[],
            )

            for i, alpha in enumerate(alpha_list):
                nums, lamhat = calibrator.calibrate(alpha, risk_func, risk_precision)
                calib_risk_results["num"] = nums
                calib_risk_results["data"].append(dict(alpha=alpha, threshold=lamhat))

                table = PrettyTable(["class", "name", "n", "threshold"])
                for cls_idx in range(self.runner.model.num_classes):
                    cls_name = calib_risk_results["name"][cls_idx]
                    table.add_row([cls_idx, cls_name, nums[cls_idx], f"{lamhat[cls_idx]:.6f}"])

                self.runner.logger.info(f"Calibration results | "
                                        f"risk: {risk_func} | "
                                        f"alpha: {alpha} | "
                                        f"size: {calibrator.num_data}\n"
                                        f"{str(table)}")

            calib_results[risk_func] = calib_risk_results

        return calib_results

    def evaluate(
            self,
            calib_results: Dict,
            dataset: BaseSegDataset,
            eval_set_idx: List[int] = None,
    ) -> Dict:
        log_interval = self.runner.cfg.default_hooks["logger"]["interval"]
        num_cls = self.runner.model.num_classes
        ignore_index = dataset.ignore_index

        if eval_set_idx is None:
            eval_set_idx = range(len(dataset))
        eval_size = len(eval_set_idx)

        calib_results = dict(calib_results)
        calib_results.pop("alpha")

        evaluators = {}
        for risk_func, calib in calib_results.items():
            evaluators[risk_func] = []
            for data in calib["data"]:
                lamhat = data["threshold"]
                evaluator = ConformalRiskEvaluator(num_cls, lamhat, ignore_index, device=get_device())
                evaluators[risk_func].append(evaluator)

        self.runner.model.eval()
        for i, idx in enumerate(eval_set_idx):
            data_batch = dataset.prepare_data(idx)
            inputs, data_samples = data_batch["inputs"], data_batch["data_samples"]
            data_batch = dict(inputs=[inputs], data_samples=[data_samples])
            seg_results = super(ConformalRisk, self).compute_iter(idx, data_batch)
            seg_logits = torch.stack([r.seg_logits.data for r in seg_results])

            probs = torch.softmax(seg_logits, dim=1)
            labels = torch.vstack([r.gt_sem_seg.data for r in seg_results]).to(get_device())

            for risk_func, evaluator_list in evaluators.items():
                for evaluator in evaluator_list:
                    evaluator.append(probs, labels)

            if i % log_interval == log_interval - 1:
                i = str(i + 1).rjust(len(str(eval_size)), " ")
                self.runner.logger.info(f"Epoch(evaluation) [{i}/{eval_size}]")

        for risk_func, evaluator_list in evaluators.items():
            for evaluator in evaluator_list:
                evaluator.done()

        eval_results = {}
        for risk_func, evaluator_list in evaluators.items():

            eval_risk_results = dict(
                risk=risk_func,
                size=eval_size,
                name=calib_results[risk_func]["name"],
                data=[],
            )

            for i, evaluator in enumerate(evaluator_list):
                alpha = calib_results[risk_func]["data"][i]["alpha"]
                nums, risks, stds = evaluator.evaluate(risk_func)

                coverage = evaluator.mean_coverage()
                length = evaluator.mean_length()
                length_pdf = evaluator.mean_length_pdf()

                eval_risk_results["num"] = nums
                eval_risk_results["data"].append(dict(
                    alpha=alpha,
                    risk=risks,
                    std=stds,
                    coverage=coverage,
                    length=length,
                    length_pdf=length_pdf,
                ))

                table = PrettyTable(["class", "name", "n", "risk", "std"])
                for cls_idx in range(num_cls):
                    cls_name = eval_risk_results["name"][cls_idx]
                    risk_entry = f"{risks[cls_idx]:.6f}" if risks[cls_idx] is not None else "-"
                    std_entry = f"{stds[cls_idx]:.6f}" if stds[cls_idx] is not None else "-"
                    table.add_row([cls_idx, cls_name, nums[cls_idx], risk_entry, std_entry])

                self.runner.logger.info(f"Evaluation results | "
                                        f"risk: {risk_func} | "
                                        f"alpha: {alpha} | "
                                        f"size: {eval_size} | "
                                        f"coverage: {coverage:.4f} | "
                                        f"length: {length:.4f}\n"
                                        f"length pdf: {[round(b, 3) for b in length_pdf]}\n"
                                        f"{str(table)}")

            eval_results[risk_func] = eval_risk_results

        return eval_results
