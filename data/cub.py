import os
from copy import deepcopy

import numpy as np
from torch.utils.data import ConcatDataset, Dataset

from config import cub_root
from data.data_utils import load_lt_split, pil_loader, validate_split

NUM_CLASSES = 200


class CUBDataset(Dataset):
    """CUB-200-2011 in its official image order; uq_idxs are the global image ids."""

    def __init__(self, root=cub_root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.img_path = []
        self.targets = []
        self.uq_idxs = []

        images = self._read_mapping('images.txt', value_type=str)
        labels = self._read_mapping('image_class_labels.txt', value_type=int)
        splits = self._read_mapping('train_test_split.txt', value_type=int)

        for image_id in sorted(images.keys()):
            if (splits[image_id] == 1) != train:
                continue
            self.img_path.append(os.path.join(root, 'images', images[image_id]))
            self.targets.append(labels[image_id] - 1)
            self.uq_idxs.append(image_id - 1)

        self.uq_idxs = np.array(self.uq_idxs)

    def _read_mapping(self, filename, value_type):
        mapping = {}
        with open(os.path.join(self.root, filename), 'r') as file:
            for line in file:
                key, value = line.strip().split(maxsplit=1)
                mapping[int(key)] = value_type(value)
        return mapping

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = pil_loader(self.img_path[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index], self.uq_idxs[index]


def subsample_dataset(dataset, idxs):
    dataset.img_path = [dataset.img_path[i] for i in idxs]
    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]
    return dataset


def get_cub_200_datasets(train_transform, test_transform, args):
    whole_training_set = CUBDataset(root=cub_root, train=True, transform=train_transform)

    split_name = f'cub200_k{args.num_labeled_classes}_imb{args.imb_ratio}'
    l_k, unl_k, unl_unk = load_lt_split(split_name)
    validate_split(l_k, unl_k, unl_unk, whole_training_set.targets,
                   args.num_labeled_classes, NUM_CLASSES, split_name)

    train_labelled = subsample_dataset(deepcopy(whole_training_set), l_k)
    unlabelled_known = subsample_dataset(deepcopy(whole_training_set), unl_k)
    unlabelled_novel = subsample_dataset(deepcopy(whole_training_set), unl_unk)

    return {
        'train_labelled': train_labelled,
        'train_unlabelled': ConcatDataset([unlabelled_known, unlabelled_novel]),
        'test': CUBDataset(root=cub_root, train=False, transform=test_transform),
    }
