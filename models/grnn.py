import pdb
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from typing import List

from data.motion.prepare_dataset import NUM_NODES


class GlstmConv(MessagePassing):
    def __init__(self, in_channels, out_channels, add_self_loops:bool=True):
        super().__init__(aggr='add')
        
        self.add_self_loops = add_self_loops
        
        self.rnn = nn.GRU(input_size=in_channels, hidden_size=out_channels, num_layers=1, 
                            bias=True, batch_first=False)
        self.bias = nn.Parameter(torch.zeros(out_channels,))
        self.reset_parameters()

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, h0=None):
        T, N, d = x.shape
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(1))

        # Step 2: Linearly transform node feature matrix.
        x, h0 = self.rnn(x, h0)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(1), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # return x, h0
        return out + self.bias, h0

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
    

class DirectMultiStepModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[64, 32], **kwargs):
        super(DirectMultiStepModel, self).__init__()
        self.__class__.__name__ = 'grnn'

        self.layer1 = GlstmConv(input_dim, hidden_dim[0])
        self.layer2 = GlstmConv(hidden_dim[0], hidden_dim[1])
        self.lin = nn.Linear(hidden_dim[1]*NUM_NODES, output_dim)

    def forward(self, x, edge_index):
        T, N, d = x.shape

        out, _ = self.layer1(x, edge_index)
        out, _ = self.layer2(out.relu(), edge_index)
        out = self.lin(out[-1].reshape(1, -1))
        return out.softmax(-1)

    def dims(self):
        return self.dimensions