#!/usr/bin/env python3
"""Run an SFDNet OBB ONNX graph with the project's custom operators.

Standard ONNX nodes run in ONNX Runtime. SelectiveScan, frequency transforms,
rotated RoIAlign, and rotated NMS call the project's existing CUDA/MMCV kernels.
This is a correctness/reference runtime, not a native TensorRT plugin package.
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
    return Path(__file__).resolve().parents[3]


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
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(_cuda(x).float(), norm="ortho"), dim=(-2, -1))
    return (
        spectrum.abs().float().cpu().numpy(),
        torch.angle(spectrum).float().cpu().numpy(),
    )


@onnx_op(
    op_type="IFFT2Shift",
    inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
    outputs=[PyCustomOpDef.dt_float],
)
def ifft2_shift(magnitude, phase):
    spectrum = torch.polar(_cuda(magnitude).float(), _cuda(phase).float())
    output = torch.fft.ifft2(
        torch.fft.ifftshift(spectrum, dim=(-2, -1)), norm="ortho")
    return output.real.float().cpu().numpy()


@onnx_op(
    op_type="RoIAlignRotated",
    inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
    outputs=[PyCustomOpDef.dt_float],
    attrs={
        "output_height": PyCustomOpDef.dt_int64,
        "output_width": PyCustomOpDef.dt_int64,
        "spatial_scale": PyCustomOpDef.dt_float,
        "sampling_ratio": PyCustomOpDef.dt_int64,
        "aligned": PyCustomOpDef.dt_int64,
        "clockwise": PyCustomOpDef.dt_int64,
    },
)
def roi_align_rotated(
    x,
    rois,
    output_height=7,
    output_width=7,
    spatial_scale=1.0,
    sampling_ratio=0,
    aligned=1,
    clockwise=0,
):
    from mmcv.ops.roi_align_rotated import RoIAlignRotatedFunction

    output = RoIAlignRotatedFunction.apply(
        _cuda(x),
        _cuda(rois),
        (int(output_height), int(output_width)),
        float(spatial_scale),
        int(sampling_ratio),
        bool(aligned),
        bool(clockwise),
    )
    return output.detach().cpu().numpy()


@onnx_op(
    op_type="RotatedNMS",
    inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
    outputs=[PyCustomOpDef.dt_int64],
    attrs={
        "iou_threshold": PyCustomOpDef.dt_float,
        "score_threshold": PyCustomOpDef.dt_float,
        "max_output": PyCustomOpDef.dt_int64,
    },
)
def rotated_nms(
    boxes,
    scores,
    iou_threshold=0.1,
    score_threshold=0.0,
    max_output=-1,
):
    from mmcv.ops import nms_rotated

    valid = np.flatnonzero(scores > float(score_threshold)).astype(np.int64)
    if valid.size == 0:
        return valid
    _, keep = nms_rotated(
        _cuda(boxes[valid]), _cuda(scores[valid]), float(iou_threshold))
    keep = keep.detach().cpu().numpy().astype(np.int64)
    if int(max_output) > 0:
        keep = keep[:int(max_output)]
    return valid[keep]


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
    mmrotate_root = detection / "mmrotate"
    for path in (detection, mmrotate_root):
        sys.path.insert(0, str(path))
    import model  # noqa: F401
    import mmrotate  # noqa: F401
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmdet.datasets import replace_ImageToTensor
    from mmdet.datasets.pipelines import Compose

    cfg = Config.fromfile(str(config))
    cfg.model.backbone.pop("pretrained", None)
    cfg.model.backbone.pop("init_cfg", None)
    pipeline = Compose(replace_ImageToTensor(cfg.data.test.pipeline))
    data = pipeline(dict(img_info=dict(filename=str(image)), img_prefix=None))
    data = collate([data], samples_per_gpu=1)
    data["img_metas"] = [item.data[0] for item in data["img_metas"]]
    data["img"] = [item.data[0] for item in data["img"]]
    data = scatter(data, ["cuda:0"])[0]
    return data["img"][0].detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Exported sfdnet-domain ONNX")
    parser.add_argument("--runtime-model", type=Path, required=True,
                        help="Output ONNX with executable ai.onnx.contrib nodes")
    parser.add_argument("--input-npy", type=Path,
                        help="Preprocessed NCHW float32 input (alternative to --image)")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--config", type=Path,
                        default=_repo_path() / "detection/mmrotate/configs/SFDNet/sodaa/SFDNet_Mamba.py")
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
