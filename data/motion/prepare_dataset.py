import pdb
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader, random_split
from utils import *


DATA_PATH = '/home/victorialena/mocap_dataset/'
# DATASET_SIZE = 100000
MAX_STEPS = 120
NUM_JOINTS = 31


def prepare_dataset(args):
    # Load data
    # Shape [num_sims, num_timesteps, num_agents, num_dims]
    features = np.load(DATA_PATH + 'features.npy', allow_pickle=True)

    _, num_timesteps, num_agents, d = features.shape
    assert MAX_STEPS == num_timesteps
    assert NUM_JOINTS == num_agents

    edges = np.block(np.load(DATA_PATH + 'edges.npy')).swapaxes(1,2)
    labels = np.load(DATA_PATH + 'labels.npy')
    
    # (maybe) normalize
    x_max = features[..., 0].max().item()
    x_min = features[..., 0].min().item()
    y_max = features[..., 1].max().item()
    y_min = features[..., 1].min().item()
    z_max = features[..., 2].max().item()
    z_min = features[..., 2].min().item()
    
    scaling = (x_max, x_min, y_max, y_min, z_max, z_min)

    if args.normalize:
        print("normalizing data...")
        features[..., 0] = normalize(features[..., 0], x_max, x_min)
        features[..., 1] = normalize(features[..., 1], y_max, y_min)
        features[..., 2] = normalize(features[..., 2], z_max, z_min)

    # Convert to pytorch cuda tensor.
    _, inverse = np.unique(labels, return_inverse=True)
    dataset = TensorDataset(torch.Tensor(features).swapaxes(1, 2), torch.tensor(edges, dtype=int), torch.tensor(inverse, dtype=int))

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

    return train_generator, val_generator, test_generator, scaling, (args.batch_size, NUM_JOINTS, MAX_STEPS, features.shape[-1])
    