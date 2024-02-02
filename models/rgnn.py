import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class GCRUCell(nn.Module):
    """
    Gated Convolutional Recurrant Unit.
    - Reset gate: $r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$
    - Update gate: $z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$
    - Candidate activation: $\hat{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)$
    - Final activation: $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \hat{h}_t$
    """
    def __init__(self, input_size, hidden_size):
        super(GCRUCell, self).__init__()

        # Reset gate
        self.xr = nn.Linear(input_size, hidden_size)
        self.hr = nn.Linear(hidden_size, hidden_size, bias=False)

        # Update gate
        self.xz = nn.Linear(input_size, hidden_size)
        self.hz = nn.Linear(hidden_size, hidden_size, bias=False)

        # New memory content
        self.xn_hn = gnn.GraphSAGE(input_size+hidden_size, hidden_size, num_layers=1, act='tanh', dropout=0.0)

    def forward(self, x, edge_index, h_prev=None):
        # pdb.set_trace()
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.xr.out_features, device=x.device)

        # Reset gate & Update gate
        r = torch.sigmoid(self.xr(x) + self.hr(h_prev))
        z = torch.sigmoid(self.xz(x) + self.hz(h_prev))

        # New memory content
        n = self.xn_hn(torch.cat([x, r * h_prev], dim=-1), edge_index)

        return (1 - z) * n + z * h_prev


class GCRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GCRU, self).__init__()
        self.gru_cell = GCRUCell(input_size, hidden_size)

    def forward(self, x, edge_index, h_prev=None):
        # x is expected to be of shape (seq_len, batch, input_size)
        outputs = []
        for t in range(x.size(0)):
            h_prev = self.gru_cell(x[t], edge_index, h_prev)
            outputs.append(h_prev)
        
        return torch.stack(outputs), h_prev


class DirectMultiStepModel(nn.Module):
    def __init__(self, input_dim, output_dim, precition_horizon, hidden_dim=[128, 64], num_layers=1):
        super(DirectMultiStepModel, self).__init__()
        
        self.precition_horizon = precition_horizon
        self.output_dim = output_dim
        
        self.layer1 = GCRU(input_dim, hidden_dim[0])
        self.layer2 = GCRU(hidden_dim[0], hidden_dim[1])
        self.fc = nn.Linear(hidden_dim[1], output_dim*precition_horizon)
        
    def forward(self, x, edge_index):
        out, _ = self.layer1(x, edge_index)
        out, hidden = self.layer2(out, edge_index)
        out = self.fc(hidden).relu()
        return out.reshape(-1, self.precition_horizon, self.output_dim).swapdims(0, 1)