import torch

from .uncertainty_map import UncertaintyMap


class ConformalRiskMap(UncertaintyMap):
    def __init__(
            self,
            pred_sets: torch.Tensor,
            valid_mask: torch.Tensor = None,
            device: str = "cpu",
    ):
        super().__init__(valid_mask, device)
        self.pred_sets = pred_sets

    def create_risk_map(self, weights: torch.Tensor = None) -> torch.Tensor:  # [batch, height, width]
        b, c, h, w = self.pred_sets.size()
        pred_sets = self.pred_sets.to(torch.float)

        if weights is None:
            weights = torch.ones((b, c), device=self.device)

        W = weights.unsqueeze(1)
        P = pred_sets.view(b, c, -1)
        WxP = torch.bmm(W, P).squeeze(1)

        risk_map = (WxP / W.sum(dim=2)).view(b, h, w)
        return risk_map

    @staticmethod
    def apply_weights(pred_sets: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:  # [batch, height, width]
        b, c, h, w = pred_sets.size()
        pred_sets = pred_sets.to(torch.float)

        W = weights.unsqueeze(1)
        P = pred_sets.view(b, c, -1)
        WxP = torch.bmm(W, P).squeeze(1)
        risk_map = (WxP / W.sum(dim=2)).view(b, h, w)
        return risk_map

    def create_cooccur_pdf(self) -> torch.Tensor:  # [batch, num_cls, num_cls]
        b, c, h, w = self.pred_sets.size()
        pred_cls = self.pred_sets.to(torch.int64).view(b, c, -1)

        cooccur_pdf = torch.zeros(b, c, c, device=self.device)
        cooccur_mat = torch.zeros(b, c, c, device=self.device, dtype=torch.int64)

        for i in range(b):
            for cls_idx in range(c):
                if self.valid_mask is not None:
                    ignore_mask = self.valid_mask[i].to(torch.int64).view(-1)
                    mask_cls = ((ignore_mask * pred_cls[i, cls_idx]) == 1)
                else:
                    mask_cls = (pred_cls[i, cls_idx] == 1)

                num_ones = mask_cls.sum()
                if num_ones > 0:
                    cooccur_mat[i, cls_idx] = pred_cls[i, :, mask_cls].sum(dim=1)

            bins = cooccur_mat[i] * cooccur_mat[i]

            cooccur_pdf[i] = bins / bins.sum(dim=1).view(-1, 1)
            cooccur_pdf[i] = torch.nan_to_num(cooccur_pdf[i], nan=0)

        return cooccur_pdf

    def compute_confusion_weights(self, cooccur_pdf: torch.Tensor) -> torch.Tensor:  # [batch, num_cls]
        num_cls = cooccur_pdf.shape[1]

        max_confusion = torch.ones(num_cls) / num_cls
        max_confusion = (-max_confusion * torch.log(max_confusion)).sum()

        confusion = torch.nansum(-cooccur_pdf * torch.log(cooccur_pdf), dim=2) / max_confusion
        return confusion
