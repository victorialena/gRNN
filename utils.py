import numpy as np
import torch


def seedall(seed:int=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def off_diag_index(n):
    return np.ravel_multi_index(np.where(np.ones((n, n)) - np.eye(n)), [n, n])


def normalize(data, data_max, data_min):
    return (data - data_min) * 2 / (data_max - data_min) - 1


def std_scaling(df):
    return (df-df.mean())/(df.std())


def minmax_scaling(df):
    return (df-df.min())/(df.max()-df.min())


def which_model(name):
    if name=='rgnn':
        from models.rgnn import DirectMultiStepModel
        return DirectMultiStepModel
    if name=='grnn':
        from models.grnn import DirectMultiStepModel
        return DirectMultiStepModel
    if name=='mlp':
        from models.baselines import mlpBaseline
        return mlpBaseline
    if name=='lstm':
        from models.baselines import lstmBaseline
        return lstmBaseline
    if name=='rnn2gnn':
        from models.baselines import rnn2gnn
        return rnn2gnn
    if name=='gnn2rnn':
        from models.baselines import gnn2rnn
        return gnn2rnn
    