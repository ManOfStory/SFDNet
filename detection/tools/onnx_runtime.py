#!/usr/bin/env python3
"""Run an SFDNet HBB ONNX graph with the project's custom operators.

The graph is copied with ``sfdnet`` nodes mapped to the ONNX Runtime Extensions
domain. Standard ONNX nodes run in ONNX Runtime; custom callbacks use the same
CUDA kernels as PyTorch, so this is a correctness/reference runtime.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime_extensions import PyCustomOpDef, get_library_path, onnx_op


def _repo_path():
    return Path(__file__).resolve().parents[2]


def _cuda(array):
    return torch.from_numpy(np.asarray(array)).cuda()


@onnx_op(
    op_type="SelectiveScan",
    inputs=[PyCustomOpDef.dt_float] * 5,
    outputs=[PyCustomOpDef.dt_float],
    attrs={"delta_softplus": PyCustomOpDef.dt_int64},
)
def selective_scan(u, delta, a, b, delta_bias, delta_softplus=1):
    import selective_scan_cuda_oflex_rh

    if b.ndim == 3:
        b = b[:, None, :, :]
    tensors = [_cuda(value) for value in (u, delta, a, b, delta_bias)]
    output, _, *_ = selective_scan_cuda_oflex_rh.fwd(
        *tensors[:4], None, tensors[4], bool(delta_softplus), 1, True
    )
    return output.detach().cpu().numpy()


@onnx_op(
    op_type="FFT2Shift",
    inputs=[PyCustomOpDef.dt_float],
    outputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
)
def fft2_shift(x):
    spectrum = torch.fft.fftshift(torch.fft.fft2(_cuda(x).float(), norm="ortho"), dim=(-2, -1))
    return spectrum.abs().float().cpu().numpy(), torch.angle(spectrum).float().cpu().numpy()


@onnx_op(
    op_type="IFFT2Shift",
    inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
    outputs=[PyCustomOpDef.dt_float],
)
def ifft2_shift(magnitude, phase):
    spectrum = torch.polar(_cuda(magnitude).float(), _cuda(phase).float())
    output = torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1)), norm="ortho")
    return output.real.float().cpu().numpy()


def make_runtime_model(source, target):
    model = onnx.load(str(source))
    changed = 0
    for node in model.graph.node:
        if node.domain == "sfdnet":
            node.domain = "ai.onnx.contrib"
            changed += 1
    if not changed:
        raise ValueError("ONNX graph contains no sfdnet custom operators")
    if not any(item.domain == "ai.onnx.contrib" for item in model.opset_import):
        model.opset_import.append(onnx.helper.make_opsetid("ai.onnx.contrib", 1))
    onnx.save(model, str(target))
    onnx.checker.check_model(model)
    return changed


def image_input(image, config):
    detection = _repo_path() / "detection"
    sys.path.insert(0, str(detection))
    import model  # noqa: F401
    from mmengine.config import Config
    from mmdet.apis.inference import get_test_pipeline_cfg
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmcv.transforms import Compose

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(config))
    cfg.model.backbone.pop("pretrained", None)
    cfg.model.backbone.pop("init_cfg", None)
    cfg.test_dataloader.dataset.pipeline = [
        step for step in cfg.test_dataloader.dataset.pipeline
        if step.get("type") not in ("LoadAnnotations", "mmdet.LoadAnnotations")
    ]
    pipeline = Compose(get_test_pipeline_cfg(cfg))
    item = pipeline(dict(img_path=str(image), img_id=0))
    preprocessor = MODELS.build(cfg.model.data_preprocessor)
    batch = preprocessor(
        dict(inputs=[item["inputs"]], data_samples=[item["data_samples"]]),
        training=False,
    )
    return batch["inputs"].detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Exported sfdnet-domain ONNX")
    parser.add_argument("--runtime-model", type=Path, required=True,
                        help="Output ONNX with executable ai.onnx.contrib nodes")
    parser.add_argument("--input-npy", type=Path,
                        help="Preprocessed NCHW float32 input (alternative to --image)")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--config", type=Path,
                        default=_repo_path() / "detection/configs/SFDNet/aitod/SFDNet_Mamba.py")
    parser.add_argument("--output-npy", type=Path)
    args = parser.parse_args()
    if bool(args.input_npy) == bool(args.image):
        parser.error("provide exactly one of --input-npy or --image")
    for path, label in ((args.model, "model"), (args.config, "config")):
        if not path.is_file():
            parser.error(f"{label} not found: {path}")
    if args.image and not args.image.is_file():
        parser.error(f"image not found: {args.image}")
    if args.input_npy and not args.input_npy.is_file():
        parser.error(f"input not found: {args.input_npy}")
    args.runtime_model.parent.mkdir(parents=True, exist_ok=True)
    count = make_runtime_model(args.model, args.runtime_model)
    if args.input_npy:
        data = np.load(args.input_npy).astype(np.float32, copy=False)
    else:
        data = image_input(args.image, args.config)
    options = ort.SessionOptions()
    options.register_custom_ops_library(get_library_path())
    session = ort.InferenceSession(
        str(args.runtime_model), options, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: data})
    if args.output_npy:
        args.output_npy.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output_npy, boxes=outputs[0], scores=outputs[1], labels=outputs[2])
    print(f"Custom operators: {count} (sfdnet -> ai.onnx.contrib)")
    print(f"Input: {tuple(data.shape)}")
    print(f"Detections: {outputs[0].shape[0]}")
    if args.output_npy:
        print(f"Outputs: {args.output_npy}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("The reference runtime requires a CUDA PyTorch build")
    main()
