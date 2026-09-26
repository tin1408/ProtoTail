"""Class-frequency estimation of BaCon, used both by BaCon's losses and by the
tail-aware capacity target of ProtoTail."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


@torch.no_grad()
def estimate_class_distribution(cl_backbone, loader, args):
    """Estimate per-class sample counts from a clustering of the training representations.

    Known classes receive the sizes of the clusters aligned with them, reordered to follow
    the ranking of their labeled counts; the remaining cluster sizes are assigned to the
    novel classes in descending order. ``loader`` iterates the whole training set with the
    test transform.
    """
    device = next(cl_backbone.parameters()).device
    est_count = torch.ones(args.num_classes).to(device)

    all_feats, targets, mask_lab = [], [], []
    for images, class_labels, _, is_labeled in loader:
        feats = F.normalize(cl_backbone(images.to(device)), dim=-1)
        all_feats.append(feats.cpu().numpy())
        targets.append(class_labels.numpy())
        mask_lab.append(is_labeled[:, 0].bool().numpy())
    all_feats = np.concatenate(all_feats)
    targets = np.concatenate(targets)
    mask_lab = np.concatenate(mask_lab)

    all_preds = torch.from_numpy(KMeans(n_clusters=args.num_classes, random_state=0).fit(all_feats).labels_)
    _, ins_num = torch.unique(all_preds, return_counts=True)

    # Align clusters with classes on the labeled samples (normalized by cluster size).
    labeled_preds = all_preds[torch.from_numpy(mask_lab)].numpy().astype(int)
    labeled_targets = targets[mask_lab].astype(int)
    w = np.zeros((args.num_classes, args.num_classes), dtype=float)
    for p, t in zip(labeled_preds, labeled_targets):
        w[p, t] += 1
    w /= ins_num[:args.num_classes].view(-1, 1).repeat(1, args.num_classes).numpy()
    _, y_true_id = linear_sum_assignment(w.max() - w)
    _, cluster_of_class = torch.from_numpy(y_true_id).sort()

    _, known_labeled_ins_num = torch.unique(torch.from_numpy(labeled_targets), return_counts=True)
    idx1 = torch.argsort(known_labeled_ins_num)
    known_cluster_ins_num = ins_num[cluster_of_class[:args.num_labeled_classes]].to(est_count.dtype).to(device)
    idx2 = torch.argsort(known_cluster_ins_num)

    est_count[:args.num_labeled_classes][idx1] = known_cluster_ins_num[idx2]
    est_count[args.num_labeled_classes:] = \
        ins_num[cluster_of_class[args.num_labeled_classes:]].sort(descending=True)[0]
    return est_count.detach()


def compute_est_adjustment(est_dist, tro):
    """Logit adjustment log(pi^tro) from the estimated class distribution."""
    freq = np.array(est_dist.cpu())
    freq = freq / freq.sum()
    return torch.from_numpy(np.log(freq ** tro + 1e-12)).to(est_dist.device)
