# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from functools import partial
from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import digit_version

from mmdet.registry import MODELS

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')
# 获取当前脚本的绝对路径
import sys
import pathlib
current_file = pathlib.Path(__file__).resolve()
# 获取 detection 目录的路径 (上两级目录)
detection_dir = current_file.parent.parent.parent
# 将 detection 目录添加到 Python 路径
if str(detection_dir) not in sys.path:
    sys.path.insert(0, str(detection_dir))
import model
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--num-images',
        type=int,
        default=100,
        help='num images of calculate model flops')
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
    parser.add_argument('--data-path', help='dataset path', type=str)
    args = parser.parse_args()
    return args

# ============================
# FPS / Latency Benchmark
# ============================
def benchmark_fps(model, data_loader, warmup=10, num_iters=50):
    model.eval()

    times = []

    with torch.no_grad():
        for idx, data_batch in enumerate(data_loader):

            if idx >= warmup + num_iters:
                break

            data = model.data_preprocessor(data_batch, True)
            inputs = data['inputs']
            data_samples = data['data_samples']

            # warmup
            if idx < warmup:
                _ = model(48, inputs, data_samples=data_samples)
                continue

            torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(48, inputs, data_samples=data_samples)

            torch.cuda.synchronize()
            end = time.perf_counter()

            times.append(end - start)

    times = np.array(times)

    latency_ms = times.mean() * 1000.0
    fps = 1.0 / times.mean()

    return fps, latency_ms


    

def inference(args, logger):
    if digit_version(torch.__version__) < digit_version('1.12'):
        logger.warning(
            'Some config files, such as configs/yolact and configs/detectors,'
            'may have compatibility issues with torch.jit when torch<1.12. '
            'If you want to calculate flops for these models, '
            'please make sure your pytorch version is >=1.12.')

    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')

    if args.data_path is not None:
        args.cfg_options = dict(
            train_dataloader=dict(dataset=dict(data_root=args.data_path)),
            val_dataloader=dict(dataset=dict(data_root=args.data_path)),
            val_evaluator=dict(ann_file=args.data_path + 'annotations/instances_val2017.json'),

        )

    cfg = Config.fromfile(args.config)
    cfg.val_dataloader.batch_size = 1
    cfg.work_dir = tempfile.TemporaryDirectory().name

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # TODO: The following usage is temporary and not safe
    # use hard code to convert mmSyncBN to SyncBN. This is a known
    # bug in mmengine, mmSyncBN requires a distributed environment，
    # this question involves models like configs/strong_baselines
    if hasattr(cfg, 'head_norm_cfg'):
        cfg['head_norm_cfg'] = dict(type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['bbox_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['mask_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)

    result = {}
    avg_flops = []
    data_loader = Runner.build_dataloader(cfg.val_dataloader)
    model = MODELS.build(cfg.model)
    model.init_weights() 
    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()
    _forward = model.forward

    if True:
        num_images = 5
        mean_flops = []
        for idx, data_batch in enumerate(data_loader):
            if idx == num_images:
                break
            data = model.data_preprocessor(data_batch, True)
            model.forward = partial(model.forward, data_samples=data['data_samples'])
            with torch.no_grad():
               out = get_model_complexity_info(model, inputs = (48, data['inputs'], ))
            params = out['params_str']
            mean_flops.append(out['flops'])
        mean_flops = np.average(np.array(mean_flops))
        print(params, mean_flops)

    print("\n===== FPS Benchmark =====")

    fps, latency = benchmark_fps(
        model,
        data_loader,
        warmup=10,
        num_iters=50
    )

    print(f"FPS: {fps:.2f}")
    print(f"Latency: {latency:.2f} ms")

    return result


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    result = inference(args, logger)


if __name__ == '__main__':
    main()
