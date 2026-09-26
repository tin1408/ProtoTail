"""Stanford Cars, read directly from the official devkit (no torchvision download).

Expected layout under CARS_ROOT:
    cars_train/*.jpg
    cars_test/*.jpg
    devkit/cars_train_annos.mat
    devkit/cars_test_annos_withlabels.mat

The Kaggle dump with anno_train.csv / anno_test.csv and class folders is also
supported. Images are ordered by filename, and uq_idxs are positional indices in
that order, which is what the precomputed splits refer to.
"""

import os
from copy import deepcopy

import numpy as np
from torch.utils.data import ConcatDataset, Dataset

from config import cars_root
from data.data_utils import load_lt_split, pil_loader, validate_split

NUM_TRAIN, NUM_CLASSES = 8144, 196


def _load_devkit_mat(mat_path):
    from scipy.io import loadmat
    annos = loadmat(mat_path, squeeze_me=True)['annotations']
    items = [(str(a['fname']), int(a['class']) - 1) for a in np.atleast_1d(annos)]
    items.sort(key=lambda x: x[0])
    return items


def _load_kaggle_csv(root, train):
    """anno_*.csv without header (filename, x1, y1, x2, y2, 1-based class)."""
    import pandas as pd
    csv_path = os.path.join(root, 'anno_train.csv' if train else 'anno_test.csv')
    df = pd.read_csv(csv_path, header=None)
    # Train and test reuse filenames, so the search is restricted to */train/* or */test/*.
    split = 'train' if train else 'test'
    path_by_name = {}
    for dirpath, _, filenames in os.walk(root):
        if split not in set(p.lower() for p in dirpath.split(os.sep)):
            continue
        for fn in filenames:
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                path_by_name.setdefault(fn, os.path.join(dirpath, fn))
    items = []
    for fname, cls in zip(df[0].astype(str), df[5].astype(int)):
        fname = os.path.basename(fname)
        if fname not in path_by_name:
            raise FileNotFoundError(f'{fname} ({csv_path}) not found under {root}.')
        items.append((path_by_name[fname], int(cls) - 1))
    items.sort(key=lambda x: os.path.basename(x[0]))
    return items


class CarsDataset(Dataset):
    def __init__(self, root=cars_root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        img_dir = os.path.join(root, 'cars_train' if train else 'cars_test')
        mat = os.path.join(root, 'devkit',
                           'cars_train_annos.mat' if train else 'cars_test_annos_withlabels.mat')
        csv = os.path.join(root, 'anno_train.csv' if train else 'anno_test.csv')
        if os.path.isfile(mat) and os.path.isdir(img_dir):
            self.samples = [(os.path.join(img_dir, f), c) for f, c in _load_devkit_mat(mat)]
        elif os.path.isfile(csv):
            self.samples = _load_kaggle_csv(root, train)
        else:
            raise FileNotFoundError(
                f'No Stanford Cars data under {root}: expected cars_train/, cars_test/ and '
                'devkit/*.mat, or anno_train.csv / anno_test.csv.')
        self.targets = [t for _, t in self.samples]
        self.uq_idxs = np.arange(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = pil_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target, int(self.uq_idxs[index])


def subsample_dataset(dataset, idxs):
    idxs = np.asarray(idxs)
    dataset.samples = [dataset.samples[i] for i in idxs]
    dataset.targets = [dataset.targets[i] for i in idxs]
    dataset.uq_idxs = np.asarray(dataset.uq_idxs)[idxs]
    return dataset


def get_stanford_cars_datasets(train_transform, test_transform, args):
    whole_training_set = CarsDataset(root=cars_root, train=True, transform=train_transform)
    assert len(whole_training_set) == NUM_TRAIN, \
        f'Stanford Cars train set has {len(whole_training_set)} images, expected {NUM_TRAIN}'

    split_name = f'cars196_k{args.num_labeled_classes}_imb{args.imb_ratio}'
    l_k, unl_k, unl_unk = load_lt_split(split_name)
    validate_split(l_k, unl_k, unl_unk, whole_training_set.targets,
                   args.num_labeled_classes, NUM_CLASSES, split_name)

    train_labelled = subsample_dataset(deepcopy(whole_training_set), l_k)
    unlabelled_known = subsample_dataset(deepcopy(whole_training_set), unl_k)
    unlabelled_novel = subsample_dataset(deepcopy(whole_training_set), unl_unk)

    return {
        'train_labelled': train_labelled,
        'train_unlabelled': ConcatDataset([unlabelled_known, unlabelled_novel]),
        'test': CarsDataset(root=cars_root, train=False, transform=test_transform),
    }
