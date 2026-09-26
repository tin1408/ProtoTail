from copy import deepcopy

from data.cub import get_cub_200_datasets
from data.data_utils import MergedDataset
from data.stanford_cars import get_stanford_cars_datasets

DATASETS = {
    # name: (loader, number of known classes, total number of classes)
    'cub200': (get_cub_200_datasets, 100, 200),
    'stanford_cars': (get_stanford_cars_datasets, 98, 196),
}


def get_class_splits(args):
    _, num_known, num_classes = DATASETS[args.dataset_name]
    args.train_classes = range(num_known)
    args.unlabeled_classes = range(num_known, num_classes)
    args.num_labeled_classes = num_known
    args.num_unlabeled_classes = num_classes - num_known
    args.num_classes = num_classes
    return args


def get_datasets(dataset_name, train_transform, test_transform, args):
    """Returns the merged train set (labeled + unlabeled) and the test set."""
    get_dataset_f = DATASETS[dataset_name][0]
    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform, args=args)
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))
    return train_dataset, datasets['test']
