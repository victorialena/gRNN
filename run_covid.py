import argparse
import pdb

import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from tqdm import trange

from data.covid.prepare_dataset import prepare_dataset, PRD_STEPS
from metrics import MetricSuite, merge_and_print
from utils import seedall, which_model

# from models.rgnn import DirectMultiStepModel
# from models.grnn import DirectMultiStepModel
# from models.baselines import lstmBaseline as DirectMultiStepModel
# from models.baselines import mlpBaseline as DirectMultiStepModel
# from models.baselines import rnn2gnn as DirectMultiStepModel
# from models.baselines import gnn2rnn as DirectMultiStepModel
from models.baselines import ZeroBaseline, RollingAvg


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--model', type=str, default='rgnn', help='Model type to be used.', choices=['rgnn', 'grnn', 'mlp', 'lstm', 'rnn2gnn', 'gnn2rnn'])
parser.add_argument('--naive', action='store_true', default=False, help='Evaluate Naive baselines.')
parser.add_argument('--num_epoch', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=1, help='Number of samples per batch.')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='Initial learning rate.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--normalize', action='store_true', default=True, help='Apply feature scaling to input data.')
parser.add_argument('--hidden_dim', nargs='+', type=int, default=[64, 64], help='List of hidden dimensions for each layer.')
parser.add_argument('--label_ids', nargs='+', type=int, default=[0, 1], help='List of hidden dimensions for each layer.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print(args)


def train(model, data_loader, optimizer, loss_fn, num_epoch):
    model.train()
    
    ep_loss = []
    for _ in trange(num_epoch, unit="Epoch"):
        losses = []
        for x, y, edges in data_loader:
            x, y, edges = x.squeeze(0), y.squeeze(0), edges.squeeze(0)
            yhat = model(x, edge_index=edges)

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
            Yhat.append(model(x, edge_index=edges))
            Y.append(y)

    Y, Yhat = torch.stack(Y, axis=0), torch.stack(Yhat, axis=0)
    return [m(Yhat, Y) for m in metrics]


def evaluateNaive():
    for model_class in [ZeroBaseline, RollingAvg]:
        model = model_class(out_channels, PRD_STEPS)

        metrics, sparse_metrics = MetricSuite(), MetricSuite(mode='sparse')
        direct_metrics = evaluate(model, test_loader, [metrics, sparse_metrics])
        merge_and_print(direct_metrics, model_class.__name__)


#-------------- MAIN

train_loader, test_loader, scaling, (_, NUM_NODES, MAX_STEPS, NUM_FEATS) = prepare_dataset(args)

in_channels, out_channels = NUM_FEATS, len(args.label_ids)

seedall()
DirectMultiStepModel = which_model(args.model)
model = DirectMultiStepModel(in_channels, out_channels, PRD_STEPS).to(device)
optimizer = Adam(model.parameters(), lr=args.learning_rate)

loss_fn = nn.MSELoss()
metrics, sparse_metrics = MetricSuite(), MetricSuite(mode='sparse')

model, loss = train(model, train_loader, optimizer, loss_fn, args.num_epoch)

direct_metrics = evaluate(model, test_loader, [metrics, sparse_metrics])
merge_and_print(direct_metrics, args.model)

if args.naive:
    evaluateNaive()
