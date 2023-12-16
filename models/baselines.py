import torch.nn as nn
import torch_geometric.nn as gnn

from collections import OrderedDict
from typing import Union, List

import pdb


class mySoftmax(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.layers = nn.Linear(in_dim, num_classes)
        self.activation = nn.Softmax()

    def forward(self, x):
        bs = x.shape[0]
        return self.activation(self.layers(x.reshape(bs, -1)))


class mlpBaseline(nn.Module):
    def __init__(self, dimensions:List[int], history:int, num_nodes:int):
        super().__init__()

        dimensions[0] *= history
        self.dimensions = dimensions
        layers = []
        
        for i, (dim_in, dim_out) in enumerate(zip(dimensions[:-2], dimensions[1:-1])):
            layers.append(('linear_'+str(i), nn.Linear(dim_in, dim_out)))
            layers.append(('relu_'+str(i), nn.ReLU()))
        self.layers = nn.Sequential(OrderedDict(layers))
        self.activation = mySoftmax(num_nodes*dimensions[-2], dimensions[-1])
        
    def forward(self, x, **kwargs):
        bs, N, T, d = x.shape
        out = self.layers(x.reshape(bs, N, T*d))
        return self.activation(out)

    def dims(self):
        return self.dimensions
    

class lstmBaseline(nn.Module):
    def __init__(self, dimensions:List[int], num_nodes:int, **kwargs):
        super().__init__()

        self.dimensions = dimensions
        layers = []
        for i, (dim_in, dim_out) in enumerate(zip(dimensions[:-2], dimensions[1:-1])):
            layers.append(('lstm_'+str(i), nn.LSTM(dim_in, dim_out, batch_first=True, dropout=0.0)))

        self.layers = nn.ModuleDict(OrderedDict(layers))
        self.activation = mySoftmax(num_nodes*dimensions[-2], dimensions[-1])
        
    def forward(self, x, **kwargs):
        bs, N, T, d = x.shape
        out, hidden = x.reshape(bs*N, T, d), None

        for _, rnn in self.layers.items():
            out, hidden = rnn(out, hidden)

        return self.activation(out[:, -1].reshape(bs, N, -1))
    
    def dims(self):
        return self.dimensions


class GNNembedding(nn.Module):
    """
    Note: self-loops are not necessary since we included them in the graph, but for good measure/std practice
    """
    def __init__(self, dimensions:List[int], node_dim=-3):
        super().__init__()
        self.dimensions = dimensions
        kwargs = {'act': 'relu', 'num_layers':1, 'dropout':0.0, 'self_loops':True, 'node_dim':node_dim}
        self.model = gnn.Sequential('x, edge_index', [ 
            (gnn.GraphSAGE(dim_in, dim_out, **kwargs), 'x, edge_index -> x') for dim_in, dim_out in zip(dimensions[:-1], dimensions[1:])
            ])
    
    def forward(self, x, edge_index):
        return self.model(x, edge_index)
    
    def dims(self):
        return self.dimensions


class gnn2rnn(nn.Module):
    def __init__(self, dimensions:List[int], **kwargs):
        super().__init__()

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
    def __init__(self, dimensions:List[int], **kwargs):
        super().__init__()

        self.dimensions = dimensions
        self.rnn = nn.LSTM(dimensions[0], dimensions[1:3], num_layers=2)
        self.gnn = GNNembedding(dimensions[3:], node_dim=-2)
        
    def forward(self, x, edge_index):
        # bs, N, T, d = x.shape
        out, hidden = self.rnn(x)
        return self.gnn(hidden, edge_index)

    def dims(self):
        return self.dimensions