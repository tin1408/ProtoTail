"""ProtoTail training on top of BaCon (long-tailed fine-grained GCD).

Example (full model on CUB, imbalance ratio 10):
    python train.py --dataset-name cub200 --exp-name cub_full \
        --use-parts --enable-pseudo-labeling --enable-novel-pseudo --use-momentum-teacher
See scripts/ for the commands of every configuration reported in the paper.
"""

import argparse
import json
import math
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import exp_root
from data.augmentations import get_transform
from data.get_datasets import get_class_splits, get_datasets
from model.backbone import (SUPPORTED_BACKBONES, backbone_spec, forward_backbone_tokens,
                            forward_blocks_from, forward_frozen_prefix, is_late_block_param,
                            load_backbone, set_finetune_blocks, split_cls_patches)
from model.distribution import compute_est_adjustment, estimate_class_distribution
from model.heads import CEHead, DINOHead, build_ce_head_like
from model.loss import (ContrastiveLearningViewGenerator, DistillLoss, SupConLoss,
                        compute_reg_loss, compute_softconloss, get_params_groups, info_nce_logits)
from model.part_modules import (LatentPartModule, PartPrototypeBank, compute_capacity_loss,
                                compute_fused_ce_loss, compute_gate_reg_loss, compute_target_capacity)
from model.pseudo_label import (NovelSelectionState, audit_pseudo_samples, balanced_sampler,
                                labeled_class_accuracy, select_known_pseudo_labels,
                                select_novel_pseudo_labels, update_train_loader, uq_to_true_label)
from util.cluster_and_log_utils import log_accs_from_preds, set_frequency_groups
from util.general_utils import AverageMeter, init_experiment

# Weights of the auxiliary part losses and prototype-memory schedule (Appendix B).
GATE_LOSS_WEIGHT = 0.05
CAPACITY_LOSS_WEIGHT = 0.05
EMA_START_EPOCH = 30
UNLABELED_EMA_END_EPOCH = 60


# ----------------------------------------------------------------------------
# Momentum teacher of the pseudo-labeling branch
# ----------------------------------------------------------------------------

def teacher_momentum(epoch, epochs, m0=0.996):
    """Cosine schedule from m0 to 1."""
    return 1.0 - (1.0 - m0) * 0.5 * (1.0 + math.cos(math.pi * epoch / max(epochs, 1)))


@torch.no_grad()
def ema_update_teacher(teacher, student, m):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)


def build_teacher(student_ce, args, device):
    try:
        teacher = deepcopy(student_ce)
    except RuntimeError:  # weight-normed modules cannot always be deep-copied
        teacher = nn.Sequential(
            deepcopy(student_ce[0]),
            build_ce_head_like(student_ce[1], args.feat_dim, args.num_classes, device)).to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher.eval()


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

@torch.no_grad()
def extract_eval_features(cl_backbone, images, part_module=None, part_bank=None):
    """Normalized CLS feature, concatenated with the gate-weighted slot feature when parts are used."""
    if part_module is None:
        return nn.functional.normalize(cl_backbone(images), dim=-1)
    cls, patch_tokens, _ = forward_backbone_tokens(cl_backbone, images)
    r_norm, _ = part_module(patch_tokens)
    g_part, _, a = part_bank(r_norm)
    a_pred = a[g_part.argmax(dim=-1)]
    r_pool = (r_norm * a_pred.unsqueeze(-1)).sum(dim=1) / (a_pred.sum(dim=1, keepdim=True) + 1e-6)
    return nn.functional.normalize(torch.cat([cls, r_pool], dim=-1), dim=-1)


@torch.no_grad()
def test(student_cl, test_loader, epoch, save_name, args, part_module=None, part_bank=None):
    """K-means on the test features, evaluated with Hungarian matching."""
    student_cl.eval()
    all_feats, targets, mask = [], [], []
    for images, label, _ in test_loader:
        feats = extract_eval_features(student_cl[0], images.to(args.device), part_module, part_bank)
        all_feats.append(feats.cpu().numpy())
        targets.append(label.numpy())
        mask.append(np.isin(label.numpy(), list(args.train_classes)))
    all_feats = np.concatenate(all_feats)
    preds = KMeans(n_clusters=args.num_classes, random_state=0).fit(all_feats).labels_
    return log_accs_from_preds(np.concatenate(targets), preds, np.concatenate(mask),
                               save_name=save_name, epoch=epoch, args=args)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train(ce_backbone, ce_head, cl_backbone, cl_head, train_loader, test_loader,
          train_all_test_trans_loader, train_labelled_test_trans_loader, args):
    logger = args.logger
    device = args.device
    set_frequency_groups(args, train_loader.dataset)

    student_ce = nn.Sequential(ce_backbone, ce_head).to(device)
    student_cl = nn.Sequential(cl_backbone, cl_head).to(device)

    def set_mode(train_mode):
        for m in (student_ce, student_cl):
            m.train(train_mode)

    teacher_ce = build_teacher(student_ce, args, device) if args.use_momentum_teacher else None

    optimizer_ce = SGD(get_params_groups(student_ce), lr=args.lr, momentum=args.momentum,
                       weight_decay=args.weight_decay)
    optimizer_cl = SGD(list(cl_head.parameters()) + list(cl_backbone.parameters()), lr=args.lr,
                       momentum=args.momentum, weight_decay=args.weight_decay)
    schedulers = [lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 1e-3)
                  for opt in (optimizer_ce, optimizer_cl)]

    # --- Prototype-guided local representation ---
    part_module = part_bank = optimizer_part = optimizer_gate = None
    if args.use_parts:
        part_module = LatentPartModule(dim=args.feat_dim, num_slots=args.num_slots).to(device)
        part_bank = PartPrototypeBank(num_classes=args.num_classes, num_slots=args.num_slots,
                                      dim=args.feat_dim).to(device)
        nn.init.constant_(part_bank.gate_logits, -1.0)  # gates start at sigmoid(-1) ~ 0.27
        optimizer_part = SGD(part_module.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
        optimizer_gate = SGD([part_bank.gate_logits], lr=0.01, momentum=0.9)
        # The part losses also update the last trainable block of the contrastive backbone.
        cl_late_params = [p for n, p in cl_backbone.named_parameters()
                          if (is_late_block_param(n, args.grad_from_block) or 'norm' in n)
                          and p.requires_grad]
        if cl_late_params:
            optimizer_part.add_param_group({'params': cl_late_params, 'lr': 0.01})
        schedulers += [lr_scheduler.CosineAnnealingLR(optimizer_part, T_max=args.epochs, eta_min=0.05 * 1e-3),
                       lr_scheduler.CosineAnnealingLR(optimizer_gate, T_max=args.epochs, eta_min=0.01 * 1e-3)]
        logger.info(f'Parts: M={args.num_slots}, tail-aware capacity={not args.ablate_adaptive_capacity}')

    # --- Dual-branch pseudo-label selection ---
    used_uq_idxs = set()
    pseudo_iteration = 0
    novel_state = NovelSelectionState()
    pseudo_events = []
    uq2true = uq_to_true_label(train_loader.dataset.unlabelled_dataset)  # diagnostics only

    cluster_criterion = DistillLoss(args.warmup_teacher_temp_epochs, args.epochs, args.n_views,
                                    args.warmup_teacher_temp, args.teacher_temp)

    set_mode(False)
    est_count = estimate_class_distribution(cl_backbone, train_all_test_trans_loader, args)
    est_adjustment = compute_est_adjustment(est_count, args.tro)

    best = {'all': -1, 'epoch': -1}
    epochs_since_best = 0
    test_history = []

    for epoch in range(args.epochs):
        set_mode(True)
        m_teacher = teacher_momentum(epoch, args.epochs, args.teacher_m0)
        loss_record_ce, loss_record_cl = AverageMeter(), AverageMeter()

        for batch_idx, (images_, class_labels, _, mask_lab) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            mask_lab = mask_lab[:, 0]
            class_labels = class_labels.to(device, non_blocking=True)
            mask_lab = mask_lab.to(device, non_blocking=True).bool()
            images = torch.cat(images_, dim=0).to(device, non_blocking=True)

            # The frozen prefix is identical for both branches and computed once.
            x = forward_frozen_prefix(ce_backbone, images, args.grad_from_block)
            ce_feature, _ = split_cls_patches(
                ce_backbone.norm(forward_blocks_from(x, ce_backbone, args.grad_from_block)), ce_backbone)
            student_out = ce_head(ce_feature)
            cl_feature, cl_patch_tokens = split_cls_patches(
                cl_backbone.norm(forward_blocks_from(x, cl_backbone, args.grad_from_block)), cl_backbone)
            cl_proj_feature = cl_head(cl_feature)

            # --- Pseudo-labeling branch (BaCon) ---
            if teacher_ce is not None:
                with torch.no_grad():
                    teacher_out = teacher_ce(images)
            else:
                teacher_out = student_out.detach()
            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
            sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
            sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
            cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
            cluster_loss += args.memax_weight * compute_reg_loss(student_out, est_count, args.p)
            loss_ce = (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss

            # --- Prototype-guided local representation (labeled samples of view 0) ---
            B_half = class_labels.shape[0] // args.n_views
            if part_module is not None:
                r_norm, _ = part_module(cl_patch_tokens)
                r_norm_v0 = r_norm[:B_half]
                mask_lab_v0 = mask_lab[:B_half]
                if mask_lab_v0.sum() > 0:
                    g_part, _, a = part_bank(r_norm_v0)
                    loss_ce = loss_ce + compute_fused_ce_loss(
                        student_out[:B_half][mask_lab_v0], g_part[mask_lab_v0],
                        class_labels[:B_half][mask_lab_v0], lambda_part=args.part_lambda, tau=args.tau_c)
                    loss_ce = loss_ce + GATE_LOSS_WEIGHT * compute_gate_reg_loss(a)
                    if not args.ablate_adaptive_capacity:
                        target = compute_target_capacity(est_count.detach().clone().float(), args.num_slots)
                        loss_ce = loss_ce + CAPACITY_LOSS_WEIGHT * compute_capacity_loss(a, target)

                for opt in (optimizer_part, optimizer_gate, optimizer_ce, optimizer_cl):
                    opt.zero_grad()
                # Gradients of loss_ce reach the contrastive branch through the parts;
                # they are accumulated with loss_cl before a single step below.
                loss_ce.backward(retain_graph=True)
            else:
                optimizer_ce.zero_grad()
                loss_ce.backward()
                optimizer_ce.step()
                if teacher_ce is not None:
                    ema_update_teacher(teacher_ce, student_ce, m_teacher)
            loss_record_ce.update(loss_ce.item(), class_labels.size(0))

            # --- Prototype memory (EMA, from epoch 30) ---
            if part_module is not None and epoch >= EMA_START_EPOCH:
                with torch.no_grad():
                    r_v0 = r_norm[:B_half].detach()
                    lab_v0, mask_v0 = class_labels[:B_half], mask_lab[:B_half]
                    if mask_v0.sum() > 0:
                        part_bank.update_ema(r_v0[mask_v0], lab_v0[mask_v0])
                    if not mask_v0.all() and epoch <= UNLABELED_EMA_END_EPOCH:
                        g_fused = student_out[:B_half] + args.part_lambda * part_bank(r_v0)[0]
                        unl = ~mask_v0
                        conf = torch.softmax(g_fused[unl] / args.tau_c, dim=-1).max(dim=-1)[0]
                        part_bank.update_ema_unlabeled(r_v0[unl], g_fused[unl].argmax(dim=-1), conf)

            # --- Contrastive branch (BaCon) ---
            cl_proj_feature = nn.functional.normalize(cl_proj_feature, dim=-1)
            contrastive_logits, contrastive_labels = info_nce_logits(features=cl_proj_feature)
            contrastive_loss = nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)
            sup_cl_proj_feature = torch.cat([f[mask_lab].unsqueeze(1) for f in cl_proj_feature.chunk(2)], dim=1)
            sup_con_labels = class_labels[mask_lab]
            sup_con_loss = SupConLoss()(sup_cl_proj_feature, labels=sup_con_labels)
            if epoch >= args.ce_warmup:
                soft_con_loss = compute_softconloss(student_out, cl_proj_feature, sup_con_labels,
                                                    sup_cl_proj_feature, mask_lab, est_count,
                                                    est_adjustment, args.alpha, args.beta)
                loss_cl = ((1 - args.sup_weight) * contrastive_loss + (args.sup_weight / 2) * sup_con_loss
                           + (args.sup_weight / 2) * soft_con_loss)
            else:
                loss_cl = (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
            loss_record_cl.update(loss_cl.item(), class_labels.size(0))

            if part_module is not None:
                loss_cl.backward()
                for opt in (optimizer_ce, optimizer_cl, optimizer_part, optimizer_gate):
                    opt.step()
                if teacher_ce is not None:
                    ema_update_teacher(teacher_ce, student_ce, m_teacher)
            else:
                optimizer_cl.zero_grad()
                loss_cl.backward()
                optimizer_cl.step()

        logger.info(f'Train epoch {epoch}: loss_ce {loss_record_ce.avg:.3f} | loss_cl {loss_record_cl.avg:.3f}')

        if (epoch + 1) % args.est_freq == 0:
            set_mode(False)
            est_count = estimate_class_distribution(cl_backbone, train_all_test_trans_loader, args)
            est_adjustment = compute_est_adjustment(est_count, args.tro)

        all_acc, old_acc, new_acc, groups, nmi, ari = test(
            student_cl, test_loader, epoch, 'Test', args, part_module, part_bank)
        test_history.append({'epoch': epoch, 'all': all_acc, 'old': old_acc, 'new': new_acc,
                             'groups': groups, 'nmi': nmi, 'ari': ari})

        # --- Dual-branch pseudo-label selection ---
        if (args.enable_pseudo_labeling and epoch >= args.pseudo_warmup_epoch
                and (epoch - args.pseudo_warmup_epoch) % args.pseudo_update_freq == 0
                and pseudo_iteration < args.max_pseudo_iterations):
            pseudo_iteration += 1
            unlab_loader = DataLoader(train_loader.dataset.unlabelled_dataset, batch_size=256,
                                      shuffle=False, num_workers=0)

            # Known branch: expand the known class with the highest labeled accuracy.
            acc = labeled_class_accuracy(student_ce, train_labelled_test_trans_loader, args.num_labeled_classes)
            target_class = int(np.argmax(acc))
            new_pseudo = select_known_pseudo_labels(student_ce, unlab_loader, target_class,
                                                    args.max_samples_per_class, used_uq_idxs,
                                                    top_ratio=args.pseudo_top_ratio)
            used_uq_idxs |= {s['uq_idx'] for s in new_pseudo}
            logger.info(f'[Known] round {pseudo_iteration}: class {target_class} '
                        f'(labeled acc {acc[target_class]:.3f}) -> {len(new_pseudo)} samples')

            # Novel branch.
            if (args.enable_novel_pseudo and epoch >= args.novel_warmup_epoch
                    and (epoch - args.novel_warmup_epoch) % args.novel_update_freq == 0
                    and novel_state.iteration < args.max_novel_iterations):
                novel_state.iteration += 1
                try:
                    new_pseudo += select_novel_pseudo_labels(
                        student_ce, cl_backbone, unlab_loader, train_loader.dataset.labelled_dataset,
                        args, novel_state, used_uq_idxs, logger)
                except Exception as e:  # a failed round is skipped, the known branch is unaffected
                    logger.warning(f'[Novel] round {novel_state.iteration} skipped: {e}')

            audit = audit_pseudo_samples(new_pseudo, uq2true, args.num_labeled_classes)
            logger.info(f'[Pseudo audit, ground truth used for logging only] {audit}')
            if new_pseudo:
                train_loader = update_train_loader(train_loader, new_pseudo)
                pseudo_events.append({'iteration': pseudo_iteration, 'epoch': epoch, **audit})

        for sch in schedulers:
            sch.step()

        # --- Checkpointing and early stopping ---
        epochs_since_best += 1
        if all_acc > best['all']:
            best = {'all': all_acc, 'old': old_acc, 'new': new_acc, 'groups': groups,
                    'nmi': nmi, 'ari': ari, 'epoch': epoch}
            epochs_since_best = 0
            torch.save({
                'epoch': epoch,
                'ce_backbone': ce_backbone.state_dict(), 'ce_head': ce_head.state_dict(),
                'cl_backbone': cl_backbone.state_dict(), 'cl_head': cl_head.state_dict(),
                'part_module': part_module.state_dict() if part_module is not None else None,
                'part_bank': part_bank.state_dict() if part_bank is not None else None,
            }, os.path.join(args.model_dir, 'best.pt'))
        if args.early_stop_patience > 0 and epochs_since_best >= args.early_stop_patience:
            logger.info(f'Early stopping at epoch {epoch} (best epoch {best["epoch"]}).')
            break

    last = test_history[-1]
    logger.info(f'Best epoch {best["epoch"]}: All {best["all"]:.2f} | Old {best["old"]:.2f} | New {best["new"]:.2f}')
    logger.info(f'Last epoch {last["epoch"]}: All {last["all"]:.2f} | Old {last["old"]:.2f} | New {last["new"]:.2f}')
    with open(os.path.join(args.log_dir, 'results.json'), 'w') as f:
        json.dump({'best': best, 'last': last, 'test_history': test_history,
                   'pseudo_events': pseudo_events}, f, indent=1)


def get_args():
    parser = argparse.ArgumentParser(description='ProtoTail on BaCon',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data and backbone
    parser.add_argument('--dataset-name', type=str, default='cub200', choices=['cub200', 'stanford_cars'])
    parser.add_argument('--imb-ratio', type=int, default=10)
    parser.add_argument('--backbone', type=str, default='dinov2_vitb14', choices=SUPPORTED_BACKBONES)
    parser.add_argument('--grad-from-block', type=int, default=11)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20364)
    parser.add_argument('--exp-root', type=str, default=exp_root)
    parser.add_argument('--exp-name', type=str, default='prototail')
    # Optimization (BaCon)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--sup-weight', type=float, default=0.35)
    parser.add_argument('--n-views', type=int, default=2)
    parser.add_argument('--memax-weight', type=float, default=1.0)
    parser.add_argument('--warmup-teacher-temp', type=float, default=0.07)
    parser.add_argument('--teacher-temp', type=float, default=0.04)
    parser.add_argument('--warmup-teacher-temp-epochs', type=int, default=50)
    parser.add_argument('--p', type=float, default=1.1, help='exponent of the distribution-aware regularizer')
    parser.add_argument('--tro', type=float, default=0.5, help='logit-adjustment exponent')
    parser.add_argument('--alpha', type=float, default=0.0, help='sampling exponent for known classes')
    parser.add_argument('--beta', type=float, default=0.5, help='sampling exponent for novel classes')
    parser.add_argument('--ce-warmup', type=int, default=1)
    parser.add_argument('--est-freq', type=int, default=10, help='epochs between distribution estimates')
    parser.add_argument('--early-stop-patience', type=int, default=50, help='0 disables early stopping')
    parser.add_argument('--use-momentum-teacher', action='store_true')
    parser.add_argument('--teacher-m0', type=float, default=0.996)
    # Prototype-guided local representation
    parser.add_argument('--use-parts', action='store_true')
    parser.add_argument('--num-slots', type=int, default=3)
    parser.add_argument('--part-lambda', type=float, default=0.5)
    parser.add_argument('--tau-c', type=float, default=0.1)
    parser.add_argument('--ablate-adaptive-capacity', action='store_true',
                        help='disable the tail-aware capacity regularizer L_cap')
    # Dual-branch pseudo-label selection
    parser.add_argument('--enable-pseudo-labeling', action='store_true', help='known-category branch')
    parser.add_argument('--pseudo-warmup-epoch', type=int, default=30)
    parser.add_argument('--pseudo-update-freq', type=int, default=10)
    parser.add_argument('--max-pseudo-iterations', type=int, default=20)
    parser.add_argument('--pseudo-top-ratio', type=float, default=0.8)
    parser.add_argument('--max-samples-per-class', type=int, default=500)
    parser.add_argument('--enable-novel-pseudo', action='store_true',
                        help='novel-category branch (requires --enable-pseudo-labeling)')
    parser.add_argument('--novel-warmup-epoch', type=int, default=50)
    parser.add_argument('--novel-update-freq', type=int, default=10)
    parser.add_argument('--max-novel-iterations', type=int, default=6)
    parser.add_argument('--novel-max-samples', type=int, default=200)
    parser.add_argument('--novel-jaccard-th', type=float, default=0.6)
    parser.add_argument('--novel-agree-th', type=float, default=0.7)
    parser.add_argument('--novel-min-size', type=int, default=10)
    return parser.parse_args()


def main():
    args = get_class_splits(get_args())
    args = init_experiment(args)
    torch.backends.cudnn.benchmark = True
    device = args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.interpolation = 3
    args.crop_pct = 0.875
    backbone = load_backbone(args.backbone)
    spec = backbone_spec(args.backbone)
    args.image_size, args.feat_dim = spec['image_size'], spec['feat_dim']
    set_finetune_blocks(backbone, args.grad_from_block)

    train_transform, test_transform = get_transform('imagenet', image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    train_dataset, test_dataset = get_datasets(args.dataset_name, train_transform, test_transform, args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Copies of the training data with the test transform (distribution estimate, labeled accuracy).
    train_all_test_trans = deepcopy(train_dataset)
    train_all_test_trans.labelled_dataset.transform = test_transform
    for ds in train_all_test_trans.unlabelled_dataset.datasets:
        ds.transform = test_transform
    train_labelled_test_trans = deepcopy(train_dataset.labelled_dataset)
    train_labelled_test_trans.transform = test_transform

    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                              shuffle=False, drop_last=True, pin_memory=True,
                              sampler=balanced_sampler(len(train_dataset.labelled_dataset),
                                                       len(train_dataset.unlabelled_dataset)))

    def eval_loader(ds):
        return DataLoader(ds, num_workers=args.num_workers, batch_size=256, shuffle=False)

    cl_head = DINOHead(in_dim=args.feat_dim, out_dim=65536, nlayers=3)
    ce_head = CEHead(in_dim=args.feat_dim, out_dim=args.num_classes)
    cl_backbone = deepcopy(backbone).to(device)
    ce_backbone = deepcopy(backbone).to(device)

    train(ce_backbone, ce_head.to(device), cl_backbone, cl_head.to(device), train_loader,
          eval_loader(test_dataset), eval_loader(train_all_test_trans),
          eval_loader(train_labelled_test_trans), args)


if __name__ == '__main__':
    main()
