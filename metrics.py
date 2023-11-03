from typing import Any
import numpy as np
import torch
import torch.nn as nn

class ADELoss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, pred, target):
        assert target.dim() == 4
        return torch.pow(target-pred, 2).sum(-2).sqrt().mean(-1)


class FDELoss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, pred, target):
        """ FDE = sqrt(dx_T^2 + dy_T^2)
            |target| = [bs, n_vars, T, d]
        """
        assert target.dim() == 4
        return torch.pow(target[:, :, -1]-pred[:, :, -1], 2).sum(-1).sqrt().mean(-1)


class MetricSuite():
    def __init__(self):
        self.mdict = {'mse': nn.MSELoss(),
                      'mae': nn.L1Loss(),
                      'ade': ADELoss(),
                      'fde': FDELoss()
                    }

    def __call__(self, pred, target):
        out = {k: fn(pred, target) for k, fn in self.mdict.item()}
        return out