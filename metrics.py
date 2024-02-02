import pdb

import numpy as np
import torch
import torch.nn as nn

from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = 1e-6
        
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y)+self.eps)
    

class mMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, yhat, y, mask):
        return (self.mse(yhat, y)*mask).sum() / mask.sum()


class ADELoss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, pred, target):
        assert target.dim() == 4
        return torch.pow(target-pred, 2).sum(-1).sqrt().mean()
    
class mADELoss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, pred, target, mask):
        assert target.dim() == 4
        return (torch.pow(target-pred, 2)*mask).sum(-1).sqrt().sum() / mask.sum()


class FDELoss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, pred, target):
        """ FDE = sqrt(dx_T^2 + dy_T^2)
            |target| = [bs, T, n_vars, d]
        """
        assert target.dim() == 4
        return torch.pow(target[:, -1]-pred[:, -1], 2).sum(-1).sqrt().mean()
    

class mACC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = BinaryAccuracy()

    def __call__(self, pred, target, mask):
        return self.fn((pred>0).to(int).cpu(), mask.to(int).cpu())
    

class mPRE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = BinaryPrecision()

    def __call__(self, pred, target, mask):
        return self.fn((pred>0).to(int).cpu(), mask.to(int).cpu())
    

class mREC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = BinaryRecall()

    def __call__(self, pred, target, mask):
        return self.fn((pred>0).to(int).cpu(), mask.to(int).cpu())


class MetricSuite():
    def __init__(self, mode='regression', num_classes=-1, device=None):
        self.mdict = None
        self.mode = mode
        if mode=='regression':
            self.mdict = {'mse': nn.MSELoss(),
                          'rse': RMSELoss(),
                          'mae': nn.L1Loss(),
                          'ade': ADELoss(),
                          'fde': FDELoss()
                        }
        elif mode=='sparse':
            self.mdict = {'acc': mACC(),
                          'rec': mREC(),
                          'pre': mPRE(),
                          'made': mMSELoss(),
                          }
        else:
            assert False, "Unknown metric mode."

    def __call__(self, pred, target):
        if self.mode == 'sparse':
            mask = target != 0
            return {k: fn(pred, target, mask) for k, fn in self.mdict.items()}
        out = {k: fn(pred, target) for k, fn in self.mdict.items()}
        return out


def print_metrics(metrics: dict):
    res = [k+": "+str(v.item()) for k, v in metrics.items()]
    print(" | ".join(res))