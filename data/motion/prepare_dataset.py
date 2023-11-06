import pdb
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader, random_split
from utils import *


DATA_PATH = 'data/motion/35/'
DATASET_SIZE = 100000
MAX_STEPS = 49
NUM_JOINTS = 31


def prepare_dataset(args):
    # Load data
    # Shape [num_sims, num_timesteps, num_agents, num_dims]
    all_feats = np.load(DATA_PATH + 'all_features.npy', allow_pickle=True)
    all_feats = all_feats[:DATASET_SIZE, :MAX_STEPS]

    all_edges = np.load(DATA_PATH + 'edges.npy').T
    if False:
        _src, _dst = np.load(DATA_PATH + 'edges.npy').T
        src, dst = np.append(_src, _dst), np.append(_dst, _src)

        tom = all_feats[0, 0, dst, :3]
        frm = all_feats[0, 0, src, :3]
        dis = torch.FloatTensor(np.sqrt(np.power(tom-frm, 2).sum(axis=-1)))

        all_edges = torch.zeros((NUM_JOINTS, NUM_JOINTS))
        all_edges[src, dst] = dis
        all_edges = torch.reshape(all_edges, [-1, NUM_JOINTS ** 2])

        # Exclude self edges
        off_diag_idx = off_diag_index(NUM_JOINTS)
        all_edges = all_edges[:, off_diag_idx]
    
    
    # (maybe) normalize
    x_max = all_feats[..., 0].max().item()
    x_min = all_feats[..., 0].min().item()
    y_max = all_feats[..., 1].max().item()
    y_min = all_feats[..., 1].min().item()
    z_max = all_feats[..., 2].max().item()
    z_min = all_feats[..., 2].min().item()
    
    scaling = (x_max, x_min, y_max, y_min, z_max, z_min)

    if args.normalize:
        print("normalizing data")
        all_feats[..., 0] = normalize(all_feats[..., 0], x_max, x_min)
        all_feats[..., 1] = normalize(all_feats[..., 1], y_max, y_min)
        all_feats[..., 2] = normalize(all_feats[..., 2], z_max, z_min)

    # Convert to pytorch cuda tensor.
    dataset = TensorDataset(torch.Tensor(all_feats).swapaxes(1, 2), torch.tensor(all_edges).repeat(all_feats.shape[0], 1, 1))

    train_size = int(len(dataset) * 0.7)
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

    return train_generator, val_generator, test_generator, scaling
    