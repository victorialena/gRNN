import pdb

import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from typing import Union, List


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
        self.xn_hn = gnn.GraphSAGE(input_size+hidden_size, hidden_size, num_layers=1, act='tanh', dropout=0.0, node_dim=-2)
        # self.xn = nn.Linear(input_dim, hidden_size)
        # self.hn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, edge_index, h_prev=None):
        if h_prev is None:
            h_prev = torch.zeros(x.size(-2), self.xr.out_features, device=x.device)

        # Reset gate & Update gate
        r = torch.sigmoid(self.xr(x) + self.hr(h_prev))
        z = torch.sigmoid(self.xz(x) + self.hz(h_prev))

        # New memory content
        n = self.xn_hn(torch.cat([x, r * h_prev], dim=-1), edge_index)
        # n = torch.tanh(self.xn(x) + self.hn(r * h_prev))

        return (1 - z) * n + z * h_prev


class GCRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GCRU, self).__init__()
        self.gru_cell = GCRUCell(input_size, hidden_size)

    def forward(self, x, edge_index, h_prev=None):
        # x is expected to be of shape (seq_len, batch, input_size)
        outputs = []
        for t in range(x.size(2)):
            h_prev = self.gru_cell(x[:, :, t], edge_index[0], h_prev)
            outputs.append(h_prev)
        
        return torch.stack(outputs, dim=2), h_prev
    

class rGNN(nn.Module):
    def __init__(self, dimensions:List[int], num_nodes, **kwargs):
        super().__init__()
        self.dimensions = dimensions
        self.num_nodes = num_nodes

        self.layer1 = GCRU(dimensions[0], dimensions[1])
        self.linear = nn.Linear(dimensions[1]*num_nodes, dimensions[2])
        self.activation = nn.Softmax()

    def forward(self, x, edge_index):
        bs, N, T, d = x.shape

        # note: no need for activation since the GRU cell has tanh activation
        out, hidden = self.layer1(x, edge_index)
        logits = self.linear(hidden.reshape(bs, -1))
        return self.activation(logits)
    
    def dims(self):
        return self.dimensions