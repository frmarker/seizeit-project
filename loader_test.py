import argparse
import sys
from typing import List

import numpy as np

try:
    from classes.config import Config
except ImportError: 
    try:
        from classes.configuration import Config  
    except Exception as exc:  
        raise ImportError(
            'Could not import Config from classes.config or classes.configuration. '
            'Please adapt the import in loader_test.py to your local config module.'
        ) from exc

from key_generator import (
    generate_data_keys_sequential,
    generate_data_keys_sequential_window,
    generate_data_keys_subsample,
)
from net.generator_ds import SegmentedGenerator, SequentialGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test data loaders and generators.')
    parser.add_argument(
        '--exp',
        type=str,
        default='eeg_hrv',
        help="Name of the experiment / configuration (e.g. 'eeg', 'eeg_hrv').",
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Which data split to test.',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='sequential',
        choices=['sequential', 'subsample', 'sequential_window'],
        help=(
            'Key generation mode: '
            "'sequential' for full sequential coverage, "
            "'subsample' for seizure-focused subsampling, "
            "'sequential_window' for fixed windows around events."
        ),
    )
    parser.add_argument(
        '--t_add',
        type=float,
        default=600.0,
        help=(
            "Half-window size in seconds for 'sequential_window' mode. "
            'Ignored for other modes.'
        ),
    )
    parser.add_argument(
        '--batch_index',
        type=int,
        default=0,
        help='Index of the batch to inspect.',
    )
    parser.add_argument(
        '--num_batches',
        type=int,
        default=1,
        help='How many consecutive batches to print from batch_index.',
    )
    parser.add_argument(
        '--shuffle',
        action="store_true",
        help='Enable shuffling in the generator (default is False).',
    )
    return parser.parse_args()


def get_recs_list(config: Config, split: str) -> List[list]:

    # Common patterns in the original SeizeIT2 code base include either
    # methods such as `get_recordings(split)` or attributes like
    # `config.train_recs`, `config.val_recs`, `config.test_recs`.

    if hasattr(config, 'get_recordings'):
        return config.get_recordings(split)  

    if split == 'train' and hasattr(config, 'train_recs'):
        return config.train_recs  
    if split == 'val' and hasattr(config, 'val_recs'):
        return config.val_recs 
    if split == 'test' and hasattr(config, 'test_recs'):
        return config.test_recs 

    raise AttributeError(
        "Could not determine recording list for split='{}'. "
        'Please adapt get_recs_list() in loader_test.py to your Config class.'.format(split)
    )


def main() -> None:
    args = parse_args()


    # Load experiment configuration
    try:
        config = Config(exp_name=args.exp)
    except TypeError:
        # Some Config implementations expect a dictionary or a path instead.
        try:
            config = Config(args.exp) 
        except Exception as exc: 
            raise RuntimeError(
                "Failed to initialise Config. Please adapt loader_test.py to "
                "match your configuration API."
            ) from exc

    print('Loaded configuration for experiment:', args.exp)

  
    # Obtain the recording list for the requested split
    recs_list = get_recs_list(config, args.split)
    print(f"Number of recordings in {args.split} split: {len(recs_list)}")

    # Generate segment keys
    if args.mode == 'sequential':
        segments = generate_data_keys_sequential(config, recs_list, verbose=True)
        GeneratorClass = SequentialGenerator
    elif args.mode == 'subsample':
        segments = generate_data_keys_subsample(config, recs_list)
        GeneratorClass = SegmentedGenerator
    else: 
        segments = generate_data_keys_sequential_window(config, recs_list, t_add=args.t_add)
        GeneratorClass = SequentialGenerator

    segments = np.asarray(segments)
    if segments.ndim != 2 or segments.shape[1] < 4:
        raise ValueError(
            'Segment keys must be an array of shape (N, 4) or compatible. '
            f"Got shape {segments.shape}."
        )

    print('Total number of segments:', len(segments))

    labels = segments[:, 3].astype(int)
    unique, counts = np.unique(labels, return_counts=True)
    print('Label distribution (label: count):')
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}")

    # Instantiate the generator and inspect a few batches
    gen = GeneratorClass(
        config=config,
        segments=segments,
        shuffle=args.shuffle,
        augment=False,
        mode=args.split,
    )

    print('Number of batches in generator:', len(gen))

    start_batch = max(0, min(args.batch_index, len(gen) - 1))
    end_batch = min(len(gen), start_batch + max(1, args.num_batches))

    for b in range(start_batch, end_batch):
        x, y = gen[b]
        print('\nBatch index:', b)
        if isinstance(x, (list, tuple)):
            for i, xi in enumerate(x):
                print(f"  x[{i}] shape:", np.shape(xi))
        else:
            print('  x shape:', np.shape(x))
        print('  y shape:', np.shape(y))

        # Print basic label stats for the batch
        y_flat = np.ravel(y)
        u_b, c_b = np.unique(y_flat, return_counts=True)
        print('  Batch label distribution:')
        for u, c in zip(u_b, c_b):
            print(f"    {u}: {c}")


if __name__ == '__main__':
    try:
        main()
    except Exception as exc: 
        print('[loader_test] ERROR:', exc, file=sys.stderr)
        raise