# ProtoTail: Prototype-Guided Representation and Pseudo-Labeling for Long-Tailed Fine-Grained Generalized Category Discovery

This repository contains the code for the long-tailed experiments of ProtoTail (Tables 1 and 2 of the paper). ProtoTail is built on top of the official implementation of [BaCon](https://github.com/JianhongBai/BaCon) and adds two components:

- **Prototype-guided local representation with tail-aware regularization** (`model/part_modules.py`): latent part slots, a class-conditioned prototype bank updated by EMA, category-specific gates with the gate regularizer `L_gate`, and the tail-aware capacity regularizer `L_cap`.
- **Dual-branch pseudo-label selection** (`model/pseudo_label.py`): a known-category branch that expands the known class with the highest labeled accuracy, and a novel-category branch that accepts clusters passing stability, compactness, agreement, size and temporal-consistency checks.

## Setup

```bash
pip install -r requirements.txt
```

The DINOv2 ViT-B/14 backbone is downloaded from `torch.hub` on first use. All experiments run on a single GPU.

## Data

Download [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) and [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), then point the environment variables to them:

```bash
export CUB_ROOT=/path/to/CUB_200_2011        # contains images/, images.txt, image_class_labels.txt, train_test_split.txt
export CARS_ROOT=/path/to/stanford_cars      # contains cars_train/, cars_test/, devkit/*.mat
```

The Kaggle version of Stanford Cars, with `anno_train.csv` and `anno_test.csv`, is also supported.

`splits/` contains the long-tailed training splits with imbalance ratio 10: 100/100 known/novel classes on CUB and 98/98 on Stanford Cars. Each split has three index arrays: `l_k` (labeled known), `unl_k` (unlabeled known) and `unl_unk` (unlabeled novel). The indices are positions in the official training set, ordered by image id for CUB and by filename for Stanford Cars. All configurations use the same splits. The test sets are the official, unchanged test splits.

## Training

Each configuration of the component analysis is one command:

| Configuration | Flags |
| --- | --- |
| Baseline (BaCon) | *(none)* |
| Dual selection | `--enable-pseudo-labeling --enable-novel-pseudo` |
| Prototype (w/o Tail-aware) | `--use-parts --use-momentum-teacher --ablate-adaptive-capacity` |
| Prototype (+Tail-aware) | `--use-parts --use-momentum-teacher` |
| Full (w/o Tail-aware) | `--use-parts --use-momentum-teacher --enable-pseudo-labeling --enable-novel-pseudo --ablate-adaptive-capacity` |
| Full (+Tail-aware) | `--use-parts --use-momentum-teacher --enable-pseudo-labeling --enable-novel-pseudo` |

For example:

```bash
python train.py --dataset-name cub200 --exp-name cub_full \
    --use-parts --use-momentum-teacher --enable-pseudo-labeling --enable-novel-pseudo
```

To run every configuration on one dataset:

```bash
bash scripts/run_cub.sh     # CUB-200-2011
bash scripts/run_cars.sh    # Stanford Cars
bash scripts/run_cub.sh full proto   # a subset of the configurations
```

The default arguments are the hyperparameters used in the paper (see `python train.py --help` and Appendix B).

## Outputs and evaluation

Each run writes to `outputs/<exp-name>_<timestamp>/` (the root can be changed with `--exp-root` or `EXP_ROOT`):

- `log.txt`: the training log.
- `results.json`: test accuracy at the best and last epochs, the per-epoch history, and the pseudo-labeling rounds. Accuracies are reported on All/Old/New classes and on the Many/Med/Few frequency groups of known and novel classes; NMI and ARI are included.
- `checkpoints/best.pt`: the weights at the best epoch.

**Evaluation.** At every epoch, the test set is clustered with K-means into as many clusters as there are classes. The clustering is evaluated with a single Hungarian matching over all test samples. The features are the `[CLS]` representation of the contrastive branch; when parts are enabled, the gate-weighted slot feature is concatenated to it.

**Model selection.** As in the BaCon implementation, the reported result is the epoch with the highest All accuracy on the test set, with early stopping after 50 epochs without improvement. The result at the last epoch is also logged and saved.

**Pseudo-label audit.** After each selection round, the log reports how many promoted samples have a correct pseudo label. This audit uses the ground-truth labels of the unlabeled training samples for logging only; they never affect selection or training.

## Code structure

```
train.py                     training loop, evaluation and arguments
model/part_modules.py        latent slots, prototype bank, L_gate, L_cap
model/pseudo_label.py        known and novel pseudo-label selection
model/distribution.py        class-frequency estimation (BaCon)
model/loss.py                contrastive, self-distillation and BaCon losses
model/heads.py, backbone.py  classifier / projection heads, DINOv2 helpers
data/                        CUB and Stanford Cars loaders
util/                        logging and clustering accuracy
splits/                      long-tailed splits (imbalance ratio 10)
```

## Acknowledgements

This code builds on [BaCon](https://github.com/JianhongBai/BaCon), [SimGCD](https://github.com/CVMI-Lab/SimGCD), [DINO](https://github.com/facebookresearch/dino) and [DINOv2](https://github.com/facebookresearch/dinov2).
