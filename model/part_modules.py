"""Prototype-guided local representation (Section 3.3 of the paper)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentPartModule(nn.Module):
    """Pools patch tokens into M shared latent slots with learnable queries."""

    def __init__(self, dim, num_slots=3, temperature=0.07):
        super().__init__()
        self.num_slots = num_slots
        self.temperature = temperature
        self.queries = nn.Parameter(torch.empty(num_slots, dim))
        nn.init.orthogonal_(self.queries)

    def forward(self, patch_tokens):
        """patch_tokens: (B, N, d) -> slot features r (B, M, d), attention (B, M, N)."""
        d = patch_tokens.shape[-1]
        logits = torch.einsum('md,bnd->bmn', self.queries, patch_tokens) / (math.sqrt(d) * self.temperature)
        attn = F.softmax(logits, dim=-1)
        r = torch.einsum('bmn,bnd->bmd', attn, patch_tokens)
        return F.normalize(r, dim=-1), attn


class PartPrototypeBank(nn.Module):
    """Class-conditioned prototypes P (C, M, d), updated by EMA only, and gates a = sigmoid(u)."""

    def __init__(self, num_classes, num_slots, dim):
        super().__init__()
        proto = torch.empty(num_classes * num_slots, dim)
        nn.init.orthogonal_(proto)
        self.register_buffer('prototypes', proto.view(num_classes, num_slots, dim))
        self.gate_logits = nn.Parameter(torch.zeros(num_classes, num_slots))

    def get_gates(self):
        return torch.sigmoid(self.gate_logits)

    def forward(self, r_norm):
        """r_norm: (B, M, d) -> part evidence g_part (B, C), similarities s (B, C, M), gates a (C, M)."""
        P = F.normalize(self.prototypes, dim=-1)
        a = self.get_gates()
        s = torch.einsum('bmd,cmd->bcm', r_norm, P)
        g_part = (s * a.unsqueeze(0)).sum(dim=-1) / (a.sum(dim=-1).unsqueeze(0) + 1e-6)
        return g_part, s, a

    @torch.no_grad()
    def update_ema(self, r_norm, labels, momentum=0.99):
        """EMA update of the prototypes of labeled samples."""
        P = self.prototypes
        for c in torch.unique(labels):
            r_c = r_norm[labels == c].mean(dim=0)
            P[c] = F.normalize(momentum * P[c] + (1 - momentum) * r_c, dim=-1)
        self.prototypes.copy_(P)

    @torch.no_grad()
    def update_ema_unlabeled(self, r_norm, pseudo_labels, conf, threshold=0.7, momentum=0.99):
        """Confidence-weighted EMA update from unlabeled samples whose confidence exceeds ``threshold``."""
        P = self.prototypes
        for c in torch.unique(pseudo_labels):
            mask = (pseudo_labels == c) & (conf > threshold)
            if mask.sum() == 0:
                continue
            w_c = conf[mask].view(-1, 1, 1)
            r_c = (r_norm[mask] * w_c).sum(dim=0) / (w_c.sum(dim=0) + 1e-6)
            P[c] = F.normalize(momentum * P[c] + (1 - momentum) * r_c, dim=-1)
        self.prototypes.copy_(P)


def compute_fused_ce_loss(g_global, g_part, labels, lambda_part=0.5, tau=0.1):
    """Cross-entropy of the fused logits (g_global + lambda_part * g_part) / tau."""
    return F.cross_entropy((g_global + lambda_part * g_part) / tau, labels)


def compute_gate_reg_loss(a, M_min=1.0, lambda_bound=0.05):
    """L_gate (Eq. 5): push gates towards {0, 1} while keeping at least M_min active slots."""
    loss_bin = (a * (1 - a)).mean()
    loss_bound = F.relu(M_min - a.sum(dim=-1)).mean()
    return loss_bin + lambda_bound * loss_bound


def compute_target_capacity(pi_hat, M):
    """kappa* (Eqs. 6-7) with a log-scale min-max mapping of the estimated frequencies."""
    pi_min, pi_max = pi_hat.min(), pi_hat.max()
    if pi_max <= pi_min:
        return torch.ones_like(pi_hat) * M
    log_pi = torch.log(pi_hat + 1e-8)
    log_min = torch.log(pi_min + 1e-8)
    log_max = torch.log(pi_max + 1e-8)
    ratio = (log_pi - log_min) / (log_max - log_min + 1e-8)
    return torch.clamp(1.0 + (M - 1.0) * ratio, 1.0, float(M))


def compute_capacity_loss(a, target_capacity):
    """L_cap (Eq. 8): squared deviation of the gate mass from the target capacity."""
    return F.mse_loss(a.sum(dim=-1), target_capacity)
