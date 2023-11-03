import argparse, os
import pdb

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader

from tqdm import trange

from utils import *
from models.rgnn import GCRU
from metrics import MetricSuite
from data.motion.prepare_dataset import prepare_dataset

DATA_PATH = 'data/motion_35'


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--num_epoch', type=int, default=150, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per batch.')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Initial learning rate.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--normalize', action='store_true', default=False, help='Apply feature scaling to input data.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, data_loader, optimizer, loss_fn, num_epoch):
    model.train()
    
    ep_loss = []
    for _ in trange(num_epoch, unit="Epoch"):
        losses = []
        
        for X, edges in data_loader:
            if args.cuda:
                X, edges = X.cuda(), edges.cuda()
            x, y = X[:, :, :-model.precition_horizon], X[:, :, -model.precition_horizon:]
            yhat = model(x, edges[0])

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
    return metrics(Yhat, Y)


class DirectMultiStepModel(nn.Module):
    def __init__(self, input_dim, output_dim, precition_horizon, hidden_dim=32):
        super(DirectMultiStepModel, self).__init__()
        
        self.precition_horizon = precition_horizon
        self.output_dim = output_dim
        
        # self.emb = GNNembedding(input_dim, hidden_dim)
        self.layer1 = GCRU(input_dim, hidden_dim)
        self.layer2 = GCRU(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim*precition_horizon)
        
    def forward(self, x, edge_index):
        # out = self.emb(x, edge_index)
        out, hidden = self.layer1(x, edge_index)
        out, _ = self.layer2(out, edge_index, hidden)
        out = self.fc(out[-1]).relu()
        return out.reshape(-1, self.precition_horizon, self.output_dim).swapdims(0, 1)


seedall(args.seed, args.cuda)
train_loader, valid_loader, test_loader, scaling = prepare_dataset(args)

model = DirectMultiStepModel(input_dim=6, output_dim=3, precition_horizon=10, hidden_dim=64).to(device)
optimizer = Adam(model.parameters(), lr=args.learning_rate)

loss_fn = nn.MSELoss()
metrics = MetricSuite()

model, loss = train(model, train_loader, optimizer, loss_fn, args.num_epoch)
direct_metrics = evaluate(model, test_loader, metrics)