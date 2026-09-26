import torch.nn as nn
from torch.nn.init import trunc_normal_


class CEHead(nn.Module):
    """Cosine classifier of the pseudo-labeling branch (weight-normalized, fixed norm)."""

    def __init__(self, in_dim, out_dim, norm_last_layer=True):
        super().__init__()
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        return self.last_layer(nn.functional.normalize(x, dim=-1, p=2))


class DINOHead(nn.Module):
    """Projection head of the contrastive branch (from DINO, Apache-2.0, Facebook, Inc.)."""

    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3,
                 hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


def build_ce_head_like(head, in_dim, out_dim, device):
    """Fresh CEHead carrying the weights of ``head`` (used to copy weight-normed modules)."""
    new_head = CEHead(in_dim=in_dim, out_dim=out_dim).to(device)
    new_head.load_state_dict(head.state_dict())
    return new_head

