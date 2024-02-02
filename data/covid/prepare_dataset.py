import pdb
import pandas as pd
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader, random_split
from utils import seedall


DATA_PATH = '/home/victorialena/gRNN/data/covid/dataset/'

PRD_STEPS = 7
MAX_STEPS = 21+PRD_STEPS
NUM_NODES = 51
NUM_FEATS = 58
TIMESTEPS = 991
SliWindow = 4


def prepare_dataset(args):
    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")

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

    if args.normalize:
        print("normalizing data...")
        features[..., 2:] = (features[..., 2:]-mu.reshape((1,1,1,-1))) / std.reshape((1,1,1,-1))
        for idx in args.label_ids:
            features[..., idx] = (features[..., idx]-features[..., idx].min()) / (features[..., idx].max()-features[..., idx].min())

    # Convert to pytorch cuda tensor.
    _input, labels = torch.Tensor(features[:, :-PRD_STEPS]), torch.Tensor(features[:, -PRD_STEPS:, :, args.label_ids])
    dataset = TensorDataset(_input.to(device), labels.to(device), edges.to(device))

    train_size = int(len(dataset) * 0.9)
    val_size = 0
    test_size = len(dataset) - train_size - val_size

    seedall()
    train_dataset, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Parameters
    params = {'batch_size': args.batch_size, 'shuffle': True}
    train_loader = DataLoader(train_dataset, **params)
    test_loader = DataLoader(test_dataset,  **params)

    return train_loader, test_loader, scaling, (args.batch_size, NUM_NODES, MAX_STEPS, NUM_FEATS)
    