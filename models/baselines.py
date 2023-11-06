import torch.nn as nn
import torch_geometric.nn as gnn

from collections import OrderedDict


class mlpBaseline(nn.Module):
    def __init__(self, dimensions:list[int]):
        self.dimensions = dimensions

        layers = []
        for i, dim_in, dim_out in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(('linear_'+str(i), nn.Linear(dim_in, dim_out)))
            layers.append(('relu_'+str(i), nn.ReLU()))
        self.layers = nn.Sequential(OrderedDict(layers))
        
    def forward(self, x, **kwargs):
        bs, N, T, d = x.shape
        return self.layers(x.reshape(bs, N, T*d))

    def dims(self):
        return self.dimensions
    

class lstmBaseline(nn.Module):
    def __init__(self, dimensions:list[int]):
        self.dimensions = dimensions
        self.layers = [nn.LSTM(dim_in, dim_out, batch_first=True, dropout=0.0) for dim_in, dim_out in zip(dimensions[:-1], dimensions[1:])]
        self.activation = nn.ReLU()
        
    def forward(self, x, **kwargs):
        bs, N, T, d = x.shape
        out, hidden = x.reshape(bs*N, T, d), None
        for rnn in self.layers:
            out, hidden = rnn(input, hidden)
            out = self.activation(out)
        return out[:, -1].reshape(bs, N, -1)
    
    def dims(self):
        return self.dimensions


class GNNembedding(nn.Module):
    """
    Note: self-loops are not necessary since we included them in the graph, but for good measure/std practice
    """
    def __init__(self, dimensions:list[int], node_dim=-3):
        super().__init__()
        self.dimensions = dimensions
        self.model = gnn.Sequential('x, edge_index', [ 
            (gnn.GraphSAGE(dim_in, dim_out, act='relu', dropout=0.0, self_loops=True, node_dim=node_dim), 'x, edge_index -> x') for dim_in, dim_out in zip(dimensions[:-1], dimensions[1:])
            ])
    
    def forward(self, x, edge_index):
        return self.model(x, edge_index)
    
    def dims(self):
        return self.dimensions


class gnn2rnn(nn.Module):
    def __init__(self, dimensions:list[int]):
        self.dimensions = dimensions
        self.gnn = GNNembedding(dimensions[:3])
        self.rnn = lstmBaseline(dimensions[3:])
        
    def forward(self, x, edge_index):
        # bs, N, T, d = x.shape
        out = self.gnn(x, edge_index)
        return self.rnn(out)

    def dims(self):
        return self.dimensions
    

class rnn2gnn(nn.Module):
    def __init__(self, dimensions:list[int]):
        self.dimensions = dimensions
        self.gnn = lstmBaseline(dimensions[:3])
        self.rnn = GNNembedding(dimensions[3:], node_dim=-2)
        
    def forward(self, x, edge_index):
        # bs, N, T, d = x.shape
        out = self.rnn(x)
        return self.gnn(out, edge_index)

    def dims(self):
        return self.dimensions