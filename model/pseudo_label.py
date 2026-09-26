"""Dual-branch pseudo-label selection (Section 3.4 of the paper).

Selection uses only model predictions, training representations and the labels of the
labeled known-class samples. Ground-truth labels of unlabeled samples are used solely by
``audit_pseudo_samples`` to log the precision of the promoted samples; they never affect
training.
"""

from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_samples
from torch.utils.data import DataLoader, Dataset

from data.data_utils import MergedDataset


def _first_view(images):
    return images[0] if isinstance(images, (list, tuple)) else images


def _device_of(module):
    return next(module.parameters()).device


@torch.no_grad()
def _predict(student_ce, images):
    probs = F.softmax(student_ce[1](F.normalize(student_ce[0](images), dim=-1)), dim=1)
    return probs.max(dim=1)


# ----------------------------------------------------------------------------
# Known-category branch
# ----------------------------------------------------------------------------

@torch.no_grad()
def labeled_class_accuracy(student_ce, labelled_loader, num_labeled):
    """Per-class accuracy of the classifier on labeled known-class samples."""
    device = _device_of(student_ce)
    student_ce.eval()
    correct, total = np.zeros(num_labeled), np.zeros(num_labeled)
    for batch in labelled_loader:
        images, labels = _first_view(batch[0]).to(device), torch.as_tensor(batch[1])
        _, preds = _predict(student_ce, images)
        for p, t in zip(preds.cpu().tolist(), labels.tolist()):
            if 0 <= t < num_labeled:
                total[t] += 1
                correct[t] += int(p == t)
    return np.where(total > 0, correct / np.maximum(total, 1), 0.0)


@torch.no_grad()
def select_known_pseudo_labels(student_ce, unlabeled_loader, target_class, max_samples,
                               used_uq_idxs, top_ratio=0.8):
    """Most confident unused unlabeled samples predicted as ``target_class``."""
    device = _device_of(student_ce)
    student_ce.eval()
    candidates = []
    for batch in unlabeled_loader:
        images = _first_view(batch[0]).to(device)
        confs, preds = _predict(student_ce, images)
        for img, pred, conf, uq in zip(images, preds, confs, batch[2]):
            uq = int(uq)
            if uq not in used_uq_idxs and pred.item() == target_class:
                candidates.append({'image': img.cpu(), 'label': pred.item(),
                                   'confidence': conf.item(), 'uq_idx': uq})
    candidates.sort(key=lambda s: s['confidence'], reverse=True)
    return candidates[:min(int(len(candidates) * top_ratio), max_samples)]


# ----------------------------------------------------------------------------
# Novel-category branch
# ----------------------------------------------------------------------------

class NovelSelectionState:
    """State carried across novel-selection rounds (temporal-consistency checks)."""

    def __init__(self):
        self.iteration = 0
        self.prev_dbi = None
        self.dim_members = {}  # novel output dimension -> uq_idxs promoted to it


def _jaccard(a, b):
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


@torch.no_grad()
def select_novel_pseudo_labels(student_ce, cl_backbone, unlab_loader, labelled_dataset, args,
                               state, used_uq_idxs, logger):
    """Cluster-based novel selection with stability, compactness, agreement and size checks."""
    device = _device_of(cl_backbone)
    num_known, num_clusters = int(args.num_labeled_classes), int(args.num_classes)
    student_ce.eval()
    cl_backbone.eval()

    # Representations (contrastive branch) and predictions (classifier) of unlabeled samples.
    U_feats, U_pred, U_conf, U_uq, U_imgs = [], [], [], [], []
    for batch in unlab_loader:
        images = _first_view(batch[0]).to(device)
        U_feats.append(F.normalize(cl_backbone(images), dim=-1).cpu())
        conf, pred = _predict(student_ce, images)
        U_pred.append(pred.cpu())
        U_conf.append(conf.cpu())
        U_uq.extend(int(x) for x in batch[2])
        U_imgs.extend(im.cpu() for im in _first_view(batch[0]))
    U_feats = torch.cat(U_feats).numpy()
    U_pred = torch.cat(U_pred).numpy().astype(int)
    U_conf = torch.cat(U_conf).numpy()
    U_uq = np.array(U_uq)
    N = len(U_feats)

    # Two independent partitions (fitted on a subsample, assigned to all samples).
    rng = np.random.RandomState(state.iteration + 12345)
    fit_idx = rng.choice(N, size=min(N, 15000), replace=False)
    km0 = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(U_feats[fit_idx])
    km1 = KMeans(n_clusters=num_clusters, random_state=1, n_init=10).fit(U_feats[fit_idx])
    cent0 = km0.cluster_centers_
    lab0 = ((U_feats[:, None, :] - cent0[None, :, :]) ** 2).sum(-1).argmin(axis=1)
    lab1 = ((U_feats[:, None, :] - km1.cluster_centers_[None, :, :]) ** 2).sum(-1).argmin(axis=1)

    # Exclude clusters matched to known classes via labeled nearest-centroid assignment.
    lab_loader = DataLoader(labelled_dataset, batch_size=256, shuffle=False, num_workers=0)
    L_feats, L_true = [], []
    for batch in lab_loader:
        images = _first_view(batch[0]).to(device)
        L_feats.append(F.normalize(cl_backbone(images), dim=-1).cpu())
        L_true.extend(int(x) for x in torch.as_tensor(batch[1]))
    L_feats = torch.cat(L_feats).numpy()
    L_assign = ((L_feats[:, None, :] - cent0[None, :, :]) ** 2).sum(-1).argmin(axis=1)
    w = np.zeros((num_clusters, num_known), dtype=int)
    for c, t in zip(L_assign, L_true):
        if 0 <= t < num_known:
            w[c, t] += 1
    rows, _ = linear_sum_assignment(w.max() - w)
    known_clusters = set(int(r) for r in rows)
    candidates = [c for c in range(num_clusters) if c not in known_clusters]

    mem0 = {c: set(U_uq[lab0 == c].tolist()) for c in range(num_clusters)}
    mem1 = {c: set(U_uq[lab1 == c].tolist()) for c in range(num_clusters)}

    # Compactness (Silhouette) and the global Davies-Bouldin circuit breaker.
    sub_idx = rng.choice(N, size=min(N, 2000), replace=False)
    sil = silhouette_samples(U_feats[sub_idx], lab0[sub_idx])
    sil_mean = {}
    for c in range(num_clusters):
        m = lab0[sub_idx] == c
        sil_mean[c] = float(sil[m].mean()) if m.any() else -1.0
    sil_med = float(np.median(list(sil_mean.values())))
    dbi = float(davies_bouldin_score(U_feats[sub_idx], lab0[sub_idx]))
    if state.prev_dbi is not None and dbi > state.prev_dbi * 1.05:
        logger.info(f'[Novel] DBI {dbi:.3f} is >5% worse than the previous round '
                    f'({state.prev_dbi:.3f}); skipping this round.')
        return []
    state.prev_dbi = dbi

    selected = []
    for c in candidates:
        members = np.where(lab0 == c)[0]
        size = len(members)
        pred_novel = U_pred[members][U_pred[members] >= num_known]
        if len(pred_novel) == 0:
            continue
        vals, counts = np.unique(pred_novel, return_counts=True)
        top_dim = int(vals[counts.argmax()])
        stability = max(_jaccard(mem0[c], mem1[b]) for b in range(num_clusters))
        agreement = float(counts.max() / size)
        if (size < args.novel_min_size or stability < args.novel_jaccard_th
                or sil_mean.get(c, -1.0) < sil_med or agreement < args.novel_agree_th):
            continue
        prev_members = state.dim_members.get(top_dim, set())
        if prev_members and _jaccard(set(U_uq[members].tolist()), prev_members) < 0.3:
            continue  # no-remap constraint

        kept, new_members = 0, set()
        for idx in members[np.argsort(-U_conf[members])]:
            if kept >= args.novel_max_samples:
                break
            uq = int(U_uq[idx])
            if uq in used_uq_idxs:
                continue
            used_uq_idxs.add(uq)
            new_members.add(uq)
            selected.append({'image': U_imgs[idx], 'label': top_dim,
                             'confidence': float(U_conf[idx]), 'uq_idx': uq})
            kept += 1
        state.dim_members[top_dim] = new_members

    logger.info(f'[Novel] {len(candidates)} candidate clusters, Silhouette median {sil_med:.3f}, '
                f'DBI {dbi:.3f} -> {len(selected)} samples selected')
    return selected


# ----------------------------------------------------------------------------
# Supervised pool update
# ----------------------------------------------------------------------------

class PseudoDataset(Dataset):
    """Promoted samples; the stored (augmented) view is returned twice as the two views."""

    def __init__(self, samples):
        self.images = [s['image'] for s in samples]
        self.labels = [s['label'] for s in samples]
        self.uq_idxs = [s['uq_idx'] for s in samples]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        return [img, img.clone()], self.labels[idx], self.uq_idxs[idx]


class CombinedLabeledDataset(Dataset):
    def __init__(self, ds1, ds2):
        self.ds1, self.ds2 = ds1, ds2

    def __len__(self):
        return len(self.ds1) + len(self.ds2)

    def __getitem__(self, idx):
        return self.ds1[idx] if idx < len(self.ds1) else self.ds2[idx - len(self.ds1)]


def balanced_sampler(label_len, unlabelled_len):
    """Sampler that balances labeled and unlabeled samples in each batch."""
    weights = [1 if i < label_len else label_len / unlabelled_len
               for i in range(label_len + unlabelled_len)]
    return torch.utils.data.WeightedRandomSampler(torch.DoubleTensor(weights),
                                                  num_samples=label_len + unlabelled_len)


def update_train_loader(train_loader, new_pseudo_samples):
    """Add promoted samples to the labeled side of the training set."""
    if not new_pseudo_samples:
        return train_loader
    train_dataset = train_loader.dataset
    labelled = CombinedLabeledDataset(train_dataset.labelled_dataset, PseudoDataset(new_pseudo_samples))
    new_train_dataset = MergedDataset(labelled_dataset=labelled,
                                      unlabelled_dataset=train_dataset.unlabelled_dataset)
    return DataLoader(new_train_dataset, batch_size=train_loader.batch_size, shuffle=False,
                      sampler=balanced_sampler(len(labelled), len(train_dataset.unlabelled_dataset)),
                      drop_last=True, pin_memory=True, num_workers=train_loader.num_workers)


# ----------------------------------------------------------------------------
# Diagnostics (ground truth is used for logging only)
# ----------------------------------------------------------------------------

def uq_to_true_label(unlabelled_dataset):
    mapping = {}
    for ds in getattr(unlabelled_dataset, 'datasets', [unlabelled_dataset]):
        for t, u in zip(ds.targets, ds.uq_idxs):
            mapping[int(u)] = int(t)
    return mapping


def audit_pseudo_samples(pseudo_samples, uq2true, num_labeled):
    """Precision of the promoted samples, per branch."""
    stats = Counter()
    for s in pseudo_samples:
        true_lbl, pseudo_lbl = uq2true[int(s['uq_idx'])], int(s['label'])
        branch = 'known' if pseudo_lbl < num_labeled else 'novel'
        stats[f'{branch}_selected'] += 1
        stats[f'{branch}_correct'] += int(true_lbl == pseudo_lbl)
    return dict(stats)
