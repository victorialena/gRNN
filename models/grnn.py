import pdb
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from typing import List


class GlstmConv(MessagePassing):
    def __init__(self, in_channels, out_channels, add_self_loops: bool = True):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        
        self.add_self_loops = add_self_loops
        
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=out_channels, num_layers=1, 
                            bias=False, batch_first=True, dropout=0.0)
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, h0=None):
        # bs, N, T, d = x.shape
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(1))

        # Step 2: Linearly transform node feature matrix.
        x, h0 = self.lstm(x.view(), h0)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(1), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        return out + self.bias, h0

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
    

class gRNN(nn.Module):
    def __init__(self, dimensions:List[int], num_nodes:int, **kwargs):
        super().__init__()
        self.dimensions = dimensions

        self.layer1 = GlstmConv(dimensions[0], dimensions[1])
        self.layer2 = GlstmConv(dimensions[1], dimensions[2])
        self.linear = nn.Linear(dimensions[2]*num_nodes, dimensions[3])
        self.activation = nn.Softmax()

    def forward(self, x, edge_index):
        bs, N, T, _ = x.shape

        out, hidden = self.layer1(x, edge_index).relu()
        out, _ = self.layer2(out, edge_index, hidden)
        out = self.linear(out[:, :, -1].reshape(bs, -1))
        return self.activation(out)

    def dims(self):
        return self.dimensions