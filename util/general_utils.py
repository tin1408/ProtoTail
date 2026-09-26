import os
from datetime import datetime

from loguru import logger


class AverageMeter:
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_experiment(args):
    """Create <exp_root>/<exp_name>_<timestamp>/ with log.txt and checkpoints/."""
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    args.log_dir = os.path.join(args.exp_root, f'{args.exp_name}_{stamp}')
    args.model_dir = os.path.join(args.log_dir, 'checkpoints')
    os.makedirs(args.model_dir, exist_ok=True)
    logger.add(os.path.join(args.log_dir, 'log.txt'))
    args.logger = logger
    logger.info(f'Experiment directory: {args.log_dir}')
    logger.info(f'Arguments: {vars(args)}')
    return args
