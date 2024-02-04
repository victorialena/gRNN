import torch
import torch.nn as nn
import torch_geometric.nn as gnn

import pdb

from data.covid.prepare_dataset import NUM_NODES, MAX_STEPS



class GNNembedding(nn.Module):
    """
    Note: self-loops are not necessary since we included them in the graph, but for good measure/std practice
    """
    def __init__(self, input_dim, output_dim, hidden_dim=[]):
        super(GNNembedding, self).__init__()
        
        dimensions = [input_dim] + hidden_dim + [output_dim]
        kwargs = {'act':'relu', 'num_layers':1, 'dropout':0.0, 'self_loops':True, 'node_dim':0}
        self.model = gnn.Sequential('x, edge_index', [ 
            (gnn.GraphSAGE(dim_in, dim_out, **kwargs), 'x, edge_index -> x') for dim_in, dim_out in zip(dimensions[:-1], dimensions[1:])
            ])
    
    def forward(self, x, edge_index):
        return self.model(x, edge_index)


class rnn2gnn(nn.Module):
    def __init__(self, input_dim, output_dim, precition_horizon, hidden_dim=[128]):
        super(rnn2gnn, self).__init__()

        self.precition_horizon = precition_horizon
        self.rnn = nn.LSTM(input_dim, hidden_dim[0], batch_first=False, dropout=0.0)
        self.gnn = GNNembedding(hidden_dim[0], output_dim*precition_horizon)
        
    def forward(self, x, edge_index):
        T, N, d = x.shape

        out, _ = self.rnn(x)
        out = self.gnn(out[-1], edge_index)
        return out.reshape(N, self.precition_horizon, -1).swapdims(0, 1)
    

class gnn2rnn(nn.Module):
    def __init__(self, input_dim, output_dim, precition_horizon, hidden_dim=[128]):
        super(gnn2rnn, self).__init__()

        self.precition_horizon = precition_horizon
        self.gnn = GNNembedding(input_dim, hidden_dim[0])
        self.rnn = nn.LSTM(hidden_dim[0], output_dim*precition_horizon, batch_first=False, dropout=0.0)
        
    def forward(self, x, edge_index):
        T, N, d = x.shape
        
        x = self.gnn(x, edge_index)
        out, _ = self.rnn(x)
        return out[-1].reshape(N, self.precition_horizon, -1).swapdims(0, 1)


class lstmBaseline(nn.Module):
    def __init__(self, input_dim, output_dim, precition_horizon, hidden_dim=[128]):
        super().__init__()
        
        self.precition_horizon = precition_horizon
        self.rnn1 = nn.LSTM(input_dim, hidden_dim[0], batch_first=False, dropout=0.0)
        self.rnn2 = nn.LSTM(hidden_dim[0], output_dim*precition_horizon, batch_first=False, dropout=0.0)
        
    def forward(self, x, **kwargs):
        T, N, d = x.shape
        
        out, _ = self.rnn1(x)
        out, _ = self.rnn2(out)
        return out[-1].reshape(N, self.precition_horizon, -1).swapdims(0, 1)


class mlpBaseline(nn.Module):
    def __init__(self, input_dim, output_dim, precition_horizon, hidden_dim=[128]):
        super(mlpBaseline, self).__init__()

        self.precition_horizon = precition_horizon
        self.layers = nn.Sequential(
            nn.Linear(input_dim*(MAX_STEPS-precition_horizon), hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], output_dim*precition_horizon),
            nn.ReLU(),
        )
        
    def forward(self, x, **kwargs):
        T, N, d = x.shape
        x = x.swapdims(0, 1).reshape(N, T*d)
        return self.layers(x).reshape(N, self.precition_horizon, -1).swapdims(0, 1)
