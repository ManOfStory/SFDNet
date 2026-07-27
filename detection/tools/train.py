# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo

import sys
import pathlib
# 获取当前脚本的绝对路径
current_file = pathlib.Path(__file__).resolve()
# 获取 detection 目录的路径 (上两级目录)
detection_dir = current_file.parent.parent
# 将 detection 目录添加到 Python 路径
if str(detection_dir) not in sys.path:
    sys.path.insert(0, str(detection_dir))

import model
import torch

from mmengine import DefaultScope
from mmengine.logging import print_log
from mmengine.utils import digit_version
import logging


import logging
import os

def build_logger(log_path="debug_stats.log"):
    logger = logging.getLogger("debug_logger")
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)

        logger.addHandler(fh)
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')

    parser.add_argument('--data-path', help='dataset path', type=str)
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def stat_tensor(name, x, logger):
    if not isinstance(x, torch.Tensor):
        return

    logger.info(
        f"{name} | mean={x.mean().item():.4e}, "
        f"std={x.std().item():.4e}, "
        f"max={x.abs().max().item():.4e}"
    )

def forward_hook(name, logger):
    def hook(module, input, output):
        if isinstance(input, tuple):
            stat_tensor(f"[FWD input] {name}", input[0], logger)
        if isinstance(output, torch.Tensor):
            stat_tensor(f"[FWD output] {name}", output, logger)
    return hook

def backward_hook(name, logger):
    def hook(module, grad_input, grad_output):
        if isinstance(grad_output[0], torch.Tensor):
            stat_tensor(f"[BWD grad output] {name}", grad_output[0], logger)
    return hook

def main():
    args = parse_args()
    #replace the data path in the config file with the data path from the command line
    if args.data_path is not None:
        args.cfg_options = dict(
            train_dataloader=dict(dataset=dict(data_root=args.data_path)),
            val_dataloader=dict(dataset=dict(data_root=args.data_path)),
            val_evaluator=dict(ann_file=args.data_path + 'annotations/instances_val2017.json'),

        )
    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume
    
    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    runner.train()


if __name__ == '__main__':
    main()
