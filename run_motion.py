import argparse, os
import pdb

import torch
import torch.nn as nn

from torch.optim import Adam
# from torch.utils.data import DataLoader

from tqdm import trange

from utils import *
from metrics import MetricSuite, print_metrics
from models.direct_multi_step import get_model, DirectMultiStepModel

from data.motion.prepare_dataset import prepare_dataset
DATA_PATH = 'data/motion_35'


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--model', type=str, default='rgnn', help='Model type to be used.', choices=['rgnn', 'mlp', 'lstm', 'rnn2gnn', 'gnn2rnn'])
parser.add_argument('--num_epoch', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--precition_horizon', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per batch.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--normalize', action='store_true', default=True, help='Apply feature scaling to input data.')
parser.add_argument('--hidden_dim', nargs='+', type=int, default=[64, 64], help='List of hidden dimensions for each layer.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args)


def train(model, data_loader, optimizer, loss_fn, num_epoch):
    model.train()
    print("Training...")
    
    ep_loss = []
    #for _ in trange(num_epoch, unit="Epoch"):
    for _ in range(num_epoch):
        losses = []
        
        for X, edges, labels in data_loader:
            if args.cuda:
                X, edges, labels = X.cuda(), edges.cuda(), labels.cuda()
            yhat = model(X, edge_index=edges)

            optimizer.zero_grad()
            loss = loss_fn(yhat, labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        print('CEloss:', np.mean(losses))
        
        ep_loss.append(np.mean(losses))
    return model, ep_loss


def evaluate(model, data_loader, metrics):
    model.eval()
    print("Testing...")
    
    Y, Yhat = [], []
    with torch.no_grad():
        for X, edges, labels in data_loader:
            if args.cuda:
                X, edges, labels = X.cuda(), edges.cuda(), labels.cuda()
            Yhat.append(model(X, edge_index=edges))
            Y.append(labels)
    Y, Yhat = torch.cat(Y, dim=0), torch.cat(Yhat, dim=0)
    return metrics(Yhat, Y)


seedall(args.seed, args.cuda)
train_loader, valid_loader, test_loader, scaling, size = prepare_dataset(args)

_, num_nodes, horizon, input_dim = size
output_dim = 4

model = get_model(args.model, dimensions=[input_dim]+args.hidden_dim+[output_dim], 
                  history=horizon-args.precition_horizon,
                  num_nodes=num_nodes)
model.to(device)
optimizer = Adam(model.parameters(), lr=args.learning_rate)

loss_fn = nn.CrossEntropyLoss()
metrics = MetricSuite(mode='classification', num_classes=output_dim, device=device)

model, loss = train(model, train_loader, optimizer, loss_fn, args.num_epoch)
direct_metrics = evaluate(model, test_loader, metrics)
print_metrics(direct_metrics)