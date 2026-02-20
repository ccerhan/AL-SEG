import torch


class UncertaintyMap(object):
    def __init__(
            self,
            valid_mask: torch.Tensor = None,
            device: str = "cpu",
    ):
        self.valid_mask = valid_mask
        self.device = device

    def compute_mean_uncertainty(self, uncert_map: torch.Tensor) -> torch.Tensor:  # [batch, 1]
        if self.valid_mask is not None:
            b, h, w = self.valid_mask.size()
            values = []
            for i in range(b):
                value = uncert_map[i, self.valid_mask[i]].mean(dim=0)
                values.append(value)
            return torch.vstack(values)
        else:
            return uncert_map.mean(dim=(1, 2)).unsqueeze(1)
