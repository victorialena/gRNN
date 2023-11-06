import numpy as np
import torch


def seedall(seed:int, cuda: bool=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def off_diag_index(n):
    return np.ravel_multi_index(np.where(np.ones((n, n)) - np.eye(n)), [n, n])


def normalize(data, data_max, data_min):
    return (data - data_min) * 2 / (data_max - data_min) - 1

