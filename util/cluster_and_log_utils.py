import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def get_dataset_targets(dataset):
    if hasattr(dataset, 'targets'):
        return np.asarray(dataset.targets, dtype=int).tolist()
    if hasattr(dataset, 'datasets'):
        return [t for child in dataset.datasets for t in get_dataset_targets(child)]
    if hasattr(dataset, 'labelled_dataset'):
        return get_dataset_targets(dataset.labelled_dataset) + get_dataset_targets(dataset.unlabelled_dataset)
    return []


def _frequency_groups(targets):
    """Rank classes by training count and split them evenly into Many / Med / Few (as in BaCon)."""
    cls_idx, ins_num = torch.unique(targets, return_counts=True)
    val, _ = torch.sort(ins_num, descending=True)
    many_thre = val[int(1 / 3 * len(cls_idx))]
    few_thre = val[int(2 / 3 * len(cls_idx))]
    many = cls_idx[ins_num > many_thre]
    few = cls_idx[ins_num < few_thre]
    med = cls_idx[(ins_num <= many_thre) & (ins_num >= few_thre)]
    return many, med, few


def set_frequency_groups(args, train_dataset):
    """Many / Med / Few groups of known and novel classes, from the training-set counts."""
    targets = torch.tensor(get_dataset_targets(train_dataset))
    known = targets[targets < args.num_labeled_classes]
    novel = targets[targets >= args.num_labeled_classes]
    args.known_groups = _frequency_groups(known)
    args.novel_groups = _frequency_groups(novel)
    args.logger.info('Known Many/Med/Few: {}/{}/{} classes, Novel Many/Med/Few: {}/{}/{} classes'.format(
        *[len(g) for g in args.known_groups], *[len(g) for g in args.novel_groups]))


def _group_acc(ind_map, w, groups):
    accs = []
    for classes in groups:
        correct = sum(w[ind_map[i], i] for i in classes.tolist())
        total = sum(w[:, i].sum() for i in classes.tolist())
        accs.append(100.0 * correct / total if total else -1.0)
    return accs


def split_cluster_acc_v2(y_true, y_pred, mask, args):
    """Clustering accuracy with a single Hungarian matching over all samples, then
    evaluated on the Old (mask=True) and New subsets and on the frequency groups."""
    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = np.vstack(linear_sum_assignment(w.max() - w)).T
    ind_map = {j: i for i, j in ind}
    total_acc = 100.0 * sum(w[i, j] for i, j in ind) / y_pred.size

    def subset_acc(classes):
        correct = sum(w[ind_map[i], i] for i in classes)
        total = sum(w[:, i].sum() for i in classes)
        return 100.0 * correct / total if total else -1.0

    group_accs = _group_acc(ind_map, w, list(args.known_groups) + list(args.novel_groups))
    return total_acc, subset_acc(old_classes_gt), subset_acc(new_classes_gt), group_accs


def log_accs_from_preds(y_true, y_pred, mask, save_name, epoch, args):
    """Returns (all, old, new, [KMany, KMed, KFew, UMany, UMed, UFew], nmi, ari)."""
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    nmi = float(normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic'))
    ari = float(adjusted_rand_score(y_true, y_pred))
    all_acc, old_acc, new_acc, g = split_cluster_acc_v2(y_true, y_pred, mask, args)
    args.logger.info(f'Epoch {epoch}, {save_name}: All {all_acc:.2f} | Old {old_acc:.2f} | New {new_acc:.2f}')
    args.logger.info(f'  KMany {g[0]:.2f} | KMed {g[1]:.2f} | KFew {g[2]:.2f} | '
                     f'UMany {g[3]:.2f} | UMed {g[4]:.2f} | UFew {g[5]:.2f} | NMI {nmi:.4f} | ARI {ari:.4f}')
    return all_acc, old_acc, new_acc, g, nmi, ari
