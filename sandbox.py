import numpy as np
import pandas as pd
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from tqdm import trange
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, random_split
from metrics import MetricSuite, print_metrics 
from data.covid.prepare_dataset import prepare_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = 'data/covid/dataset/'

PRD_STEPS = 7
MAX_STEPS = 21+PRD_STEPS
NUM_NODES = 51
NUM_FEATS = 58
TIMESTEPS = 991
SliWindow = 4
label_ids = [0, 1]


def seedall(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


#--------- PREPARE DATASET
data = pd.read_csv(DATA_PATH + 'features.csv')
data = data.sort_values(by=['date', 'location_key'], ignore_index=True)

X = data.drop(columns=['date', 'location_key']).to_numpy().reshape((TIMESTEPS, NUM_NODES, NUM_FEATS))
features = np.zeros((TIMESTEPS//SliWindow, MAX_STEPS, NUM_NODES, NUM_FEATS))
for i, idx in enumerate(range(0, TIMESTEPS-MAX_STEPS, SliWindow)):
    features[i] = X[idx:idx+MAX_STEPS]
    
assert np.isnan(features).sum() == 0


edges = np.load(DATA_PATH + 'edges.npy')
edges = torch.tensor(edges, dtype=int).repeat(TIMESTEPS//SliWindow, 1, 1)

mu, std = data.mean(numeric_only=True).to_numpy()[2:], data.std(numeric_only=True).to_numpy()[2:] 
scaling = (mu, std)

print("normalizing data...")
features[..., 2:] = (features[..., 2:]-mu.reshape((1,1,1,-1))) / std.reshape((1,1,1,-1))
for idx in label_ids:
    features[..., idx] = (features[..., idx]-features[..., idx].min()) / (features[..., idx].max()-features[..., idx].min())

# Convert to pytorch cuda tensor.
_input, labels = torch.Tensor(features[:, :-PRD_STEPS]), torch.Tensor(features[:, -PRD_STEPS:, :, label_ids])
dataset = TensorDataset(_input.to(device), labels.to(device), edges.to(device))

train_size = int(len(dataset) * 0.9)
val_size = 0
test_size = len(dataset) - train_size - val_size

seedall()
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Parameters
params = {'batch_size': 1, 'shuffle': True}
train_loader = DataLoader(train_dataset, **params)
test_loader = DataLoader(test_dataset,  **params)


#---------------- MODEL

def train(model, data_loader, optimizer, loss_fn, num_epoch):
    model.train()
    
    ep_loss = []
    for _ in trange(num_epoch, unit="Epoch"):
        losses = []
        for x, y, edges in data_loader:
            x, y, edges = x.squeeze(0), y.squeeze(0), edges.squeeze(0)
            yhat = model(x, edges)

            optimizer.zero_grad()
            loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        print('MSEloss:', np.mean(losses))
        
        ep_loss.append(np.mean(losses))
    return model, ep_loss


def evaluate(model, data_loader, metrics):
    model.eval()
    
    Y, Yhat = [], []
    with torch.no_grad():
        for x, y, edges in data_loader:
            x, y, edges = x.squeeze(0), y.squeeze(0), edges.squeeze(0)
            Yhat.append(model(x, edges))
            Y.append(y)

    Y, Yhat = torch.stack(Y, axis=0), torch.stack(Yhat, axis=0)
    return [m(Yhat, Y) for m in metrics]


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



#-------------- TRAIN

in_channels, out_channels = features.shape[-1], len(label_ids)

seedall()
model = DirectMultiStepModel(in_channels, out_channels, PRD_STEPS).to(device)
optimizer = Adam(model.parameters(), lr=2e-4)

loss_fn = nn.MSELoss()
metrics = MetricSuite()
sparse_metrics = MetricSuite(mode='sparse')


num_epoch = 10

model, loss = train(model, train_loader, optimizer, loss_fn, num_epoch)
direct_metrics = evaluate(model, test_loader, [metrics, sparse_metrics])

print_metrics(direct_metrics[0])
print_metrics(direct_metrics[1])





