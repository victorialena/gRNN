import torch.nn as nn

from models.baselines import lstmBaseline, mlpBaseline
from models.rgnn import rGNN


def get_model(name, **kwargs):
    if name=='rgnn':
        return rGNN(*kwargs)
    if name=='mlp':
        return mlpBaseline(*kwargs)
    if name=='lstm':
        return lstmBaseline(*kwargs)
    

class DirectMultiStepModel(nn.Module):
    def __init__(self, model, output_dim, precition_horizon):
        super(DirectMultiStepModel, self).__init__()
        
        self.precition_horizon = precition_horizon
        
        self.model = model
        self.fc = nn.Linear(model.dims[-1], output_dim*precition_horizon)
        
    def forward(self, x, edge_index):
        bs, N, T, d = x.shape
        out = self.model(x, edge_index=edge_index)
        out = self.fc(out)
        return out.reshape(bs, N, self.precition_horizon, -1)