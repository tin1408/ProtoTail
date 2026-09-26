import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLearningViewGenerator:
    """Returns ``n_views`` independently augmented views of an image."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]


class SupConLoss(nn.Module):
    """Supervised contrastive loss (https://arxiv.org/abs/2004.11362), from SupContrast."""

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """features: (B, n_views, d); labels: (B,)."""
        device = features.device
        batch_size, contrast_count = features.shape[:2]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_dot_contrast = torch.matmul(contrast_feature, contrast_feature.T) / self.temperature
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * contrast_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(contrast_count, batch_size).mean()


class SemiConLoss(nn.Module):
    """Contrastive loss with a soft positive mask (BaCon)."""

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, soft_mask):
        device = features.device
        batch_size, contrast_count = features.shape[:2]

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_dot_contrast = torch.matmul(contrast_feature, contrast_feature.T) / self.temperature
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        if len(logits) != len(soft_mask):
            soft_mask = soft_mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(soft_mask), 1,
                                    torch.arange(batch_size * contrast_count).view(-1, 1).to(device), 0)
        soft_mask = soft_mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (soft_mask * log_prob).sum(1) / soft_mask.sum(1)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(contrast_count, batch_size).mean()


def info_nce_logits(features, n_views=2, temperature=1.0):
    device = features.device
    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for _ in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # Discard the diagonal, then put the positive in column 0.
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    return logits, labels


def get_params_groups(model):
    """Biases and norm parameters are excluded from weight decay."""
    regularized, not_regularized = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith('.bias') or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


class DistillLoss(nn.Module):
    """Self-distillation between the two views with a warmed-up teacher temperature."""

    def __init__(self, warmup_teacher_temp_epochs, nepochs, ncrops=2, warmup_teacher_temp=0.07,
                 teacher_temp=0.04, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = (student_output / self.student_temp).chunk(self.ncrops)
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1).detach().chunk(2)

        total_loss, n_loss_terms = 0, 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue  # skip same-view pairs
                total_loss += torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1).mean()
                n_loss_terms += 1
        return total_loss / n_loss_terms


def compute_reg_loss(student_out, est_count, p):
    """BaCon's distribution-aware entropy regularizer on the mean prediction."""
    avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
    avg_probs = avg_probs * 1 / est_count ** p
    avg_probs /= avg_probs.sum()
    return -torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))


def compute_softconloss(student_out, cl_proj_feature, sup_con_labels, sup_cl_proj_feature,
                        mask_lab, est_dist, est_adjustment, alpha, beta):
    """BaCon's soft contrastive loss with frequency-aware sampling of unlabeled samples."""
    device = student_out.device
    logits = (student_out / 0.1) - est_adjustment
    view1_probs, view2_probs = logits.softmax(dim=1).chunk(2)
    soft_labels = (view1_probs + view2_probs) / 2

    known_class_sampling_rate = ((1 / est_dist) * est_dist.min()) ** alpha
    existing_class_idx, _ = torch.unique(sup_con_labels, return_counts=True)
    sampling_rate = ((1 / est_dist) * est_dist.min()) ** beta
    sampling_rate[existing_class_idx] = known_class_sampling_rate[existing_class_idx]
    batch_confidence, batch_preds = soft_labels[~mask_lab].max(dim=1)

    sampling_mask = torch.zeros(len(soft_labels)).bool().to(device)
    idx_all = torch.arange(len(soft_labels)).to(device)
    for cls_idx in torch.unique(batch_preds):
        cls_ins_num = (batch_preds == cls_idx).sum()
        cls_sample_num = torch.bernoulli(torch.tensor([sampling_rate[cls_idx]] * cls_ins_num)).sum().int()
        cls_conf = batch_confidence[batch_preds == cls_idx]
        _, idx = torch.topk(cls_conf, k=cls_sample_num)
        sampling_mask[idx_all[~mask_lab][batch_preds == cls_idx][idx]] = True

    semicon_soft_labels = soft_labels[(~mask_lab) & sampling_mask]
    semicon_feats = torch.cat([f[(~mask_lab) & sampling_mask].unsqueeze(1)
                               for f in cl_proj_feature.chunk(2)], dim=1)

    sup_cl_proj_feature = torch.cat([sup_cl_proj_feature, semicon_feats], dim=0)
    soft_targets = torch.cat([soft_labels[mask_lab], semicon_soft_labels], dim=0)
    soft_mask = (soft_targets.unsqueeze(1) * soft_targets.unsqueeze(0)).sum(dim=2)
    soft_mask[torch.eye(len(soft_mask), device=soft_mask.device).bool()] = 1

    return SemiConLoss()(sup_cl_proj_feature, soft_mask=soft_mask.detach())
