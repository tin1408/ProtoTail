import os

# Dataset roots and output directory; override with environment variables.
cub_root = os.environ.get('CUB_ROOT', 'datasets/CUB_200_2011')
cars_root = os.environ.get('CARS_ROOT', 'datasets/stanford_cars')
exp_root = os.environ.get('EXP_ROOT', 'outputs')

# Precomputed long-tailed splits (positional indices into the sorted train set).
split_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splits')
