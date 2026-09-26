"""ViT backbones from torch.hub (DINOv2 ViT-B/14 by default) and token-level helpers."""

import re

import torch

_HUB_REPO = {
    'dino_vitb16': ('facebookresearch/dino:main', 'dino_vitb16'),
    'dinov2_vitb14': ('facebookresearch/dinov2', 'dinov2_vitb14'),
    'dinov2_vitb14_reg': ('facebookresearch/dinov2', 'dinov2_vitb14_reg'),
}

_BACKBONE_SPECS = {
    'dino_vitb16': {'feat_dim': 768, 'patch_size': 16, 'image_size': 224},
    'dinov2_vitb14': {'feat_dim': 768, 'patch_size': 14, 'image_size': 224},
    'dinov2_vitb14_reg': {'feat_dim': 768, 'patch_size': 14, 'image_size': 224},
}

SUPPORTED_BACKBONES = tuple(_HUB_REPO)


def load_backbone(name='dinov2_vitb14'):
    repo, entry = _HUB_REPO[name]
    if name.startswith('dinov2_'):
        return torch.hub.load(repo, entry, pretrained=True)
    return torch.hub.load(repo, entry)


def backbone_spec(name):
    return _BACKBONE_SPECS[name]


def prepare_tokens(backbone, images):
    if hasattr(backbone, 'prepare_tokens_with_masks'):  # DINOv2
        return backbone.prepare_tokens_with_masks(images)
    return backbone.prepare_tokens(images)  # DINOv1


def iter_blocks(backbone):
    """Yield transformer blocks in order, flattening DINOv2 BlockChunk containers."""
    for blk in backbone.blocks:
        if isinstance(blk, torch.nn.ModuleList) and not hasattr(blk, 'norm1'):
            for sub in blk:
                if not isinstance(sub, torch.nn.Identity):
                    yield sub
        else:
            yield blk


def split_cls_patches(x, backbone):
    """Split (B, 1 + n_reg + N, d) tokens into the CLS token and the N patch tokens."""
    n_reg = int(getattr(backbone, 'num_register_tokens', 0) or 0)
    return x[:, 0], x[:, 1 + n_reg:]


def forward_frozen_prefix(backbone, images, grad_from_block=11):
    """Token preparation followed by the frozen blocks [0, grad_from_block)."""
    x = prepare_tokens(backbone, images)
    for i, blk in enumerate(iter_blocks(backbone)):
        if i < grad_from_block:
            x = blk(x)
    return x


def forward_blocks_from(x, backbone, start=0):
    """Run the blocks [start, end)."""
    for i, blk in enumerate(iter_blocks(backbone)):
        if i >= start:
            x = blk(x)
    return x


def forward_backbone_tokens(backbone, images):
    """Full forward pass up to the final norm; returns (cls, patches, tokens)."""
    x = backbone.norm(forward_blocks_from(prepare_tokens(backbone, images), backbone))
    cls, patches = split_cls_patches(x, backbone)
    return cls, patches, x


_BLOCK_IDX_RE = re.compile(r'\.(\d+)(?=\.|$)')


def parse_block_index(param_name):
    """Block index of a parameter name (``blocks.<i>.*`` or chunked ``blocks.<c>.<i>.*``)."""
    if 'block' not in param_name:
        return None
    matches = _BLOCK_IDX_RE.findall(param_name)
    return int(matches[-1]) if matches else None


def is_late_block_param(param_name, late_index=11):
    if f'block.{late_index}' in param_name or f'blocks.{late_index}' in param_name:
        return True
    return parse_block_index(param_name) == late_index


def set_finetune_blocks(backbone, grad_from_block=11):
    """Freeze the backbone except blocks >= grad_from_block (the final norm stays frozen)."""
    for p in backbone.parameters():
        p.requires_grad = False
    for name, p in backbone.named_parameters():
        idx = parse_block_index(name)
        if idx is not None and idx >= grad_from_block:
            p.requires_grad = True
