# flake8: noqa
import sys
import os.path as osp
import torch
import yaml
from basicsr.train import train_pipeline

import gfpgan.archs
import gfpgan.data
import gfpgan.models


def _resolve_opt_path(argv):
    for idx, arg in enumerate(argv):
        if arg == '-opt' and idx + 1 < len(argv):
            return argv[idx + 1]
    raise RuntimeError('Missing required argument `-opt <yaml>`.')


def _load_requested_num_gpu(opt_path):
    with open(opt_path, 'r', encoding='utf-8') as f:
        opt = yaml.safe_load(f)
    requested = opt.get('num_gpu', 'auto')
    if requested == 'auto':
        requested = torch.cuda.device_count()
    requested = int(requested)
    return requested


def _enforce_gpu_only():
    opt_path = _resolve_opt_path(sys.argv[1:])
    requested_num_gpu = _load_requested_num_gpu(opt_path)
    if requested_num_gpu <= 0:
        raise RuntimeError(
            f'Invalid `num_gpu={requested_num_gpu}` in {opt_path}. '
            'GFPGAN training is GPU-only; num_gpu must be > 0.'
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA is not available. GFPGAN training cannot fall back to CPU. '
            'Please fix CUDA runtime/driver and rerun.'
        )
    visible_count = torch.cuda.device_count()
    if visible_count < requested_num_gpu:
        raise RuntimeError(
            f'Only {visible_count} CUDA device(s) visible, but config requests {requested_num_gpu}. '
            'Check CUDA_VISIBLE_DEVICES and training config.'
        )


if __name__ == '__main__':
    _enforce_gpu_only()
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
