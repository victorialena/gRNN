import pdb

import numpy as np
import torch
import torch.nn as nn

from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassRecall, MulticlassPrecision


class ADELoss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, pred, target):
        assert target.dim() == 4
        return torch.pow(target-pred, 2).sum(-1).sqrt().mean()


class FDELoss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, pred, target):
        """ FDE = sqrt(dx_T^2 + dy_T^2)
            |target| = [bs, n_vars, T, d]
        """
        assert target.dim() == 4
        return torch.pow(target[..., -1, :]-pred[..., -1, :], 2).sum(-1).sqrt().mean()


class MetricSuite():
    def __init__(self, mode='regression', num_classes=-1, device=None):
        self.mdict = None
        if mode=='regression':
            self.mdict = {'mse': nn.MSELoss(),
                          'mae': nn.L1Loss(),
                          'ade': ADELoss(),
                          'fde': FDELoss()
                        }
        elif mode=='classification':
            assert num_classes > 1
            self.mdict = {'cel': nn.CrossEntropyLoss(),
                          'mf1': MulticlassF1Score(num_classes).to(device),
                          'acc': MulticlassAccuracy(num_classes).to(device),
                          'pre': MulticlassPrecision(num_classes).to(device),
                          'rec': MulticlassRecall(num_classes).to(device)
                        }

        else:
            assert False, "Unknown metric mode."

    def __call__(self, pred, target):
        out = {k: fn(pred, target) for k, fn in self.mdict.items()}
        return out
    
    
def print_metrics(metrics: dict):
    res = [k+": "+str(v.item()) for k, v in metrics.items()]
    print(" | ".join(res))