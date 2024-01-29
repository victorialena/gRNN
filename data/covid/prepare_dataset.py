import pdb
import pandas as pd
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader, random_split
from utils import *


DATA_PATH = '/home/victorialena/gRNN/data/covid/dataset/'
# DATASET_SIZE = 100000
MAX_STEPS = 30+10
NUM_NODES = 51
NUM_FEATS = 58
TIMESTEPS = 991
SliWindow = 4


def prepare_dataset(args):
    # Load data
    # Shape [num_sims, num_timesteps, num_agents, num_dims]
    features = pd.read_csv(DATA_PATH + 'features.csv', header=True)
    data = data.sort_values(by=['date', 'location_key'], ignore_index=True)
    
    X = data.drop(['date', 'location_key']).to_numpy().reshape((TIMESTEPS, NUM_NODES, NUM_FEATS))
    features = np.zeros((TIMESTEPS//SliWindow, MAX_STEPS, NUM_NODES, NUM_FEATS))
    for i, idx in enumerate(range(0, TIMESTEPS-SliWindow, SliWindow)):
        features[i] = X[idx:idx+MAX_STEPS]

    assert np.isnan(features).sum() == 0

    edges = np.load(DATA_PATH + 'edges.npy')
    
    # (maybe) normalize
    mu, std = data.mean(numeric_only=True).to_numpy(), data.std(numeric_only=True).to_numpy()    
    scaling = (mu, std)

    if args.normalize:
        print("normalizing data...(via std. scaling)")
        features = (features-mu.reshape((1,1,1,-1))) / std.reshape((1,1,1,-1))

    # Convert to pytorch cuda tensor.
    input, labels = features[:, :-10], features[:, -10:]    
    dataset = TensorDataset(torch.Tensor(input).swapaxes(1, 2), 
                            torch.tensor(edges, dtype=int), 
                            torch.Tensor(labels).swapaxes(1, 2))

    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - val_size

    
    seedall(args.seed, args.cuda)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Parameters
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': 1,
              'pin_memory': False,
              }
    
    train_generator = DataLoader(train_dataset, **params)
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_generator, val_generator, test_generator, scaling, (args.batch_size, NUM_NODES, MAX_STEPS, NUM_FEATS)
    