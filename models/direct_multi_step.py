import torch.nn as nn
import torch_geometric.nn as gnn


class GNNembedding(nn.Module):
    """
    Note: self-loops are not necessary since we included them in the graph, but for good measure/std practice
    """
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.model = gnn.Sequential('x, edge_index', [
            (gnn.GraphSAGE(input_dim, output_dim, num_layers=1, act='relu', dropout=0.0), 'x, edge_index -> x'),
        ])
    
    def forward(self, x, edge_index):
        return self.model(x, edge_index)

