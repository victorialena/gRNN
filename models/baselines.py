import torch
import torch.nn as nn
import torch_geometric.nn as gnn

import pdb

from data.motion.prepare_dataset import NUM_NODES, MAX_STEPS, PRD_STEPS 

RA_WINDOW = 10


class GNNembedding(nn.Module):
    """
    Note: self-loops are not necessary since we included them in the graph, but for good measure/std practice
    """
    def __init__(self, input_dim, output_dim, hidden_dim=[]):
        super(GNNembedding, self).__init__()
        
        dimensions = [input_dim] + hidden_dim + [output_dim]
        kwargs = {'act':'relu', 'num_layers':1, 'dropout':0.0, 'self_loops':True, 'node_dim':-2}
        self.model = gnn.Sequential('x, edge_index', [ 
            (gnn.GraphSAGE(dim_in, dim_out, **kwargs), 'x, edge_index -> x') for dim_in, dim_out in zip(dimensions[:-1], dimensions[1:])
            ])
    
    def forward(self, x, edge_index):
        return self.model(x, edge_index)


class rnn2gnn(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[64, 64]):
        super(rnn2gnn, self).__init__()

        self.rnn = nn.LSTM(input_dim, hidden_dim[0], batch_first=False, dropout=0.0)
        self.gnn = GNNembedding(hidden_dim[0], hidden_dim[1])
        self.lin = nn.Linear(hidden_dim[1]*NUM_NODES, output_dim)
        
    def forward(self, x, edge_index):
        # T, N, d = x.shape

        out, _ = self.rnn(x)
        out = self.gnn(out[-1], edge_index)
        out = self.lin(out.reshape(1, -1))
        return out.softmax(-1)
    

class gnn2rnn(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[64, 64]):
        super(gnn2rnn, self).__init__()

        self.gnn = gnn.GraphSAGE(input_dim, hidden_dim[0], num_layers=1)
        self.rnn = nn.LSTM(hidden_dim[0], hidden_dim[1], batch_first=False, dropout=0.0)
        self.lin = nn.Linear(hidden_dim[1]*NUM_NODES, output_dim)
        
    def forward(self, x, edge_index):
        # T, N, d = x.shape
        
        x = self.gnn(x, edge_index)
        out, _ = self.rnn(x)
        out = self.lin(out[-1].reshape(1, -1))
        return out.softmax(-1)
    

class lstmBaseline(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[32, 32]):
        super().__init__()
        
        self.rnn1 = nn.LSTM(input_dim, hidden_dim[0], batch_first=False, dropout=0.0)
        self.rnn2 = nn.LSTM(hidden_dim[0], hidden_dim[1], batch_first=False, dropout=0.0)
        self.lin1 = nn.Linear(hidden_dim[1]*NUM_NODES, output_dim)
        
    def forward(self, x, **kwargs):
        # T, N, d = x.shape
        
        out, _ = self.rnn1(x)
        out, _ = self.rnn2(out)
        out = self.lin1(out[-1].reshape(1, -1))
        return out.softmax(-1)


class mlpBaseline(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[1024, 1024]):
        super(mlpBaseline, self).__init__()

        self.embed = nn.Linear(input_dim*(MAX_STEPS-PRD_STEPS), hidden_dim[0])
        self.layers = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim[0]*NUM_NODES, hidden_dim[1]),
            nn.Tanh(),
            nn.Linear(hidden_dim[1], output_dim),
            nn.Softmax(-1),
        )
        
    def forward(self, x, **kwargs):
        T, N, d = x.shape
        out = x.swapdims(0, 1).reshape(N, T*d)
        out = self.embed(out)
        return self.layers(out.reshape(1, -1))
    
    
class ZeroBaseline(nn.Module):
    def __init__(self, output_dim, precition_horizon):
        super(ZeroBaseline, self).__init__()

        self.precition_horizon = precition_horizon
        self.output_dim = output_dim
        
    def forward(self, x, **kwargs):
        T, N, d = x.shape
        return torch.zeros((self.precition_horizon, N, self.output_dim), device=x.device)
    

class RollingAvg(nn.Module):
    def __init__(self, output_dim, precition_horizon):
        super(RollingAvg, self).__init__()

        self.precition_horizon = precition_horizon
        self.output_dim = output_dim
        
    def forward(self, x, **kwargs):
        T, N, _ = x.shape
        y = torch.zeros((T+self.precition_horizon, N, self.output_dim), device=x.device)
        y[:T] = x[..., :self.output_dim]
        for t in range(T, T+self.precition_horizon):
            y[t] = y[t-RA_WINDOW:t].mean(0)
        return y[-self.precition_horizon:]
    

class ConstantAvg(nn.Module):
    def __init__(self, output_dim, precition_horizon):
        super(ConstantAvg, self).__init__()

        self.precition_horizon = precition_horizon
        self.output_dim = output_dim
        
    def forward(self, x, **kwargs):
        T, N, _ = x.shape
        avg = x[..., :self.output_dim].mean(0)
        return avg.repeat(self.precition_horizon, 1, 1)
