import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from config import split_root


def pil_loader(path):
    with open(path, 'rb') as f:
        return Image.open(f).convert('RGB')


def load_lt_split(split_name):
    """Load the labeled-known / unlabeled-known / unlabeled-novel index arrays."""
    split_dir = os.path.join(split_root, split_name)
    return tuple(np.load(os.path.join(split_dir, f'{part}.npy'))
                 for part in ('l_k', 'unl_k', 'unl_unk'))


def validate_split(l_k, unl_k, unl_unk, targets, num_known, num_classes, name):
    """Check that the split indices are disjoint and respect the known/novel partition."""
    t = np.asarray(targets)
    for part, idx in (('l_k', l_k), ('unl_k', unl_k), ('unl_unk', unl_unk)):
        assert idx.max() < len(t), f'{name}: {part} index out of range'
    parts = [set(l_k.tolist()), set(unl_k.tolist()), set(unl_unk.tolist())]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2]), \
        f'{name}: split files overlap'
    known = set(int(c) for c in t[np.concatenate([l_k, unl_k])])
    novel = set(int(c) for c in t[unl_unk])
    assert known.isdisjoint(novel), f'{name}: known/novel classes overlap'
    assert len(known) == num_known, f'{name}: {len(known)} known classes != {num_known}'
    assert len(novel) == num_classes - num_known, \
        f'{name}: {len(novel)} novel classes != {num_classes - num_known}'


class MergedDataset(Dataset):
    """Concatenates a labeled and an unlabeled dataset and flags labeled items."""

    def __init__(self, labelled_dataset, unlabelled_dataset):
        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):
        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1
        else:
            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0
        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)
