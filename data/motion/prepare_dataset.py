import pdb
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader, random_split
from utils import normalize, seedall


# DATA_PATH = '/home/victorialena/mocap_dataset/'
DATA_PATH = '/home/victorialena/dGVAE/data/motion/35/'

MAX_STEPS = 49
NUM_NODES = 31
NUM_FEATS = 6
PRD_STEPS = 10



def prepare_dataset(args):
    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")

    # Load data
    features = np.load(DATA_PATH + 'all_features.npy', allow_pickle=True)
    features = features[:, :MAX_STEPS]

    _src, _dst = np.load(DATA_PATH + 'edges.npy').T
    src, dst = torch.tensor(np.append(_src, _dst)), torch.tensor(np.append(_dst, _src))
    edges = torch.stack([src, dst])
    
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
    _input, labels = torch.Tensor(features[:, :-PRD_STEPS]), torch.Tensor(features[:, -PRD_STEPS:, :, args.label_ids])
    dataset = TensorDataset(_input.to(device), labels.to(device), edges.repeat(_input.shape[0], 1, 1).to(device))

    train_size = int(len(dataset) * 0.9)
    val_size = 0
    test_size = len(dataset) - train_size - val_size
    
    seedall(args.seed)
    
    train_dataset, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader, scaling, (args.batch_size, NUM_NODES, MAX_STEPS, NUM_FEATS)
    