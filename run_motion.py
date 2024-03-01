import argparse
import pdb

import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from tqdm import trange

from data.motion.prepare_dataset import prepare_dataset, PRD_STEPS
from metrics import MetricSuite, merge_and_print
from utils import seedall, which_model, unnormalize


from models.baselines import RollingAvg, ConstantAvg


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--model', type=str, default='rgnn', help='Model type to be used.', choices=['rgnn', 'grnn', 'mlp', 'lstm', 'rnn2gnn', 'gnn2rnn'])
parser.add_argument('--num_epoch', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--naive', action='store_true', default=False, help='Evaluate Naive baselines.')
parser.add_argument('--batch_size', type=int, default=1, help='Number of samples per batch.')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='Initial learning rate.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--normalize', action='store_true', default=True, help='Apply feature scaling to input data.')
parser.add_argument('--hidden_dim', nargs='+', type=int, default=[64, 64], help='List of hidden dimensions for each layer.')
parser.add_argument('--label_ids', nargs='+', type=int, default=[0, 1, 2], help='List of hidden dimensions for each layer.')


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
        print('MSELoss:', np.mean(losses))
        
        ep_loss.append(np.mean(losses))
    return model, ep_loss


def evaluate(model, data_loader, metrics, save=False, naive=False):
    model.eval()
    
    Y, Yhat = [], []
    with torch.no_grad():
        for x, y, edges in data_loader:
            x, y, edges = x.squeeze(0), y.squeeze(0), edges.squeeze(0)
            Yhat.append(model(x, edge_index=edges))
            Y.append(y)

    Y, Yhat = torch.stack(Y, axis=0), torch.stack(Yhat, axis=0)
    if args.normalize:
        print("unnormalizing data...")
        Y[..., 0] = unnormalize(Y[..., 0], x_max, x_min)
        Y[..., 1] = unnormalize(Y[..., 1], y_max, y_min)
        Y[..., 2] = unnormalize(Y[..., 2], z_max, z_min)
        Yhat[..., 0] = unnormalize(Yhat[..., 0], x_max, x_min)
        Yhat[..., 1] = unnormalize(Yhat[..., 1], y_max, y_min)
        Yhat[..., 2] = unnormalize(Yhat[..., 2], z_max, z_min)
    if save:
        torch.save(torch.stack([Y, Yhat], dim=0), 'tmp/logs/'+model.__class__.__name__+'.pt')
    return [m(Yhat, Y) for m in metrics]


def evaluateNaive():
    for model_class in [ConstantAvg, RollingAvg]:
        model = model_class(out_channels, PRD_STEPS)

        metrics = MetricSuite()
        direct_metrics = evaluate(model, test_loader, [metrics], naive=True)
        merge_and_print(direct_metrics, model_class.__name__)


#-------------- MAIN

train_loader, test_loader, scaling, (_, NUM_NODES, MAX_STEPS, NUM_FEATS) = prepare_dataset(args)
if args.normalize:
    x_max, x_min, y_max, y_min, z_max, z_min = scaling

in_channels, out_channels = NUM_FEATS, len(args.label_ids)

seedall()
DirectMultiStepModel = which_model(args.model)
model = DirectMultiStepModel(in_channels, out_channels, PRD_STEPS, hidden_dim=args.hidden_dim).to(device)
optimizer = Adam(model.parameters(), lr=args.learning_rate)

loss_fn = nn.MSELoss()
metrics = MetricSuite()

model, loss = train(model, train_loader, optimizer, loss_fn, args.num_epoch)

direct_metrics = evaluate(model, test_loader, [metrics])
merge_and_print(direct_metrics, args.model)

if args.naive:
    evaluateNaive()
