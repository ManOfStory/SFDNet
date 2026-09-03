#!/usr/bin/env python3
"""Export an SFDNet HBB checkpoint to a fixed-shape ONNX graph.

The Spatial-Mamba selective scan and ASD frequency transforms are represented
as explicit ``sfdnet`` custom-domain nodes. A deployment runtime must provide
implementations for those nodes; stock ONNX Runtime cannot execute them.
"""

import argparse
import importlib
import sys
import types
from collections import Counter
from pathlib import Path

import torch
from torch.onnx import register_custom_op_symbolic, symbolic_helper


DETECTION_DIR = Path(__file__).resolve().parent.parent
if str(DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTION_DIR))

import model  # noqa: F401,E402 - registers SFDNet model components
from mmcv.transforms import Compose  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmdet.apis import init_detector  # noqa: E402
from mmdet.apis.inference import get_test_pipeline_cfg  # noqa: E402


DEFAULT_CONFIG = DETECTION_DIR / "configs/SFDNet/aitod/SFDNet_Mamba.py"
OPSET_VERSION = 13
REQUIRED_CUSTOM_OPS = {"SelectiveScan", "FFT2Shift", "IFFT2Shift"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export an SFDNet HBB checkpoint to ONNX")
    parser.add_argument("image", help="Sample image used to fix the input shape")
    parser.add_argument("checkpoint", help="Trained checkpoint path")
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), help="Model config path")
    parser.add_argument(
        "--output-file",
        default=None,
        help="ONNX path (default: outputs/<checkpoint>.onnx)")
    parser.add_argument(
        "--device", default="cuda:0", help="Export device (CUDA is required)")
    parser.add_argument(
        "--score-thr",
        type=float,
        default=None,
        help="Final score threshold (default: use config)")
    parser.add_argument(
        "--nms-thr",
        type=float,
        default=None,
        help="Final NMS IoU threshold (default: use config)")
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip ONNX checker and graph-integrity checks")
    return parser.parse_args()


def validate_file(path, label):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def configure_model(cfg, score_thr, nms_thr):
    if "pretrained" in cfg.model.backbone:
        cfg.model.backbone.pretrained = None
    if "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    test_cfg = cfg.model.test_cfg.rcnn
    if score_thr is not None:
        if not 0.0 <= score_thr <= 1.0:
            raise ValueError("--score-thr must be between 0 and 1")
        test_cfg.score_thr = score_thr
    if nms_thr is not None:
        if not 0.0 <= nms_thr <= 1.0:
            raise ValueError("--nms-thr must be between 0 and 1")
        test_cfg.nms.iou_threshold = nms_thr
    return float(test_cfg.score_thr), float(test_cfg.nms.iou_threshold)


def remove_annotation_loader(pipeline):
    return [
        step for step in pipeline
        if step.get("type") not in ("LoadAnnotations", "mmdet.LoadAnnotations")
    ]


def register_export_ops():
    """Register semantically equivalent ONNX mappings without editing models."""
    import models.spatialmamba as spatialmamba
    import models.utils as backbone_utils
    import mmdet.models.necks.asd as asd_module
    import mmdet.models.necks.utils.utils as neck_utils
    import mmdet.models.roi_heads.bbox_heads.bbox_head as bbox_head_module

    nms_module = importlib.import_module("mmcv.ops.nms")
    roi_align_module = importlib.import_module("mmcv.ops.roi_align")
    original_nms = nms_module.nms

    def expm1_symbolic(g, x):
        one = g.op("Constant", value_t=torch.tensor(1.0))
        return g.op("Sub", g.op("Exp", x), one)

    register_custom_op_symbolic("aten::expm1", expm1_symbolic, OPSET_VERSION)

    class ExportDepthwiseFunction:
        @staticmethod
        def apply(x, weight, bias, pad_h, pad_w, is_nhwc):
            if is_nhwc:
                x = x.permute(0, 3, 1, 2)
            output = torch.nn.functional.conv2d(
                x, weight, bias, padding=(pad_h, pad_w), groups=x.shape[1])
            return output.permute(0, 2, 3, 1) if is_nhwc else output

    spatialmamba.DepthwiseFunction = ExportDepthwiseFunction

    def scan_symbolic(g, u, delta, a, b, d, z, delta_bias,
                      delta_softplus, return_last_state, lag=0):
        output = g.op(
            "sfdnet::SelectiveScan", u, delta, a, b, delta_bias,
            delta_softplus_i=1)
        u_sizes = u.type().sizes()
        a_sizes = a.type().sizes()
        if u_sizes and a_sizes:
            output.setType(u.type().with_sizes(
                [u_sizes[0], u_sizes[1], a_sizes[1], u_sizes[2]]))
        return output

    backbone_utils.SelectiveScanStateFn.symbolic = staticmethod(scan_symbolic)
    neck_utils.SelectiveScanStateFn.symbolic = staticmethod(scan_symbolic)

    original_fft = asd_module.fft2_shift
    original_ifft = asd_module.ifft2_shift

    class FFT2Shift(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return original_fft(x)

        @staticmethod
        def symbolic(g, x):
            magnitude, phase = g.op("sfdnet::FFT2Shift", x, outputs=2)
            magnitude.setType(x.type())
            phase.setType(x.type())
            return magnitude, phase

    class IFFT2Shift(torch.autograd.Function):
        @staticmethod
        def forward(ctx, magnitude, phase):
            return original_ifft(magnitude, phase)

        @staticmethod
        def symbolic(g, magnitude, phase):
            output = g.op("sfdnet::IFFT2Shift", magnitude, phase)
            return output.setType(magnitude.type())

    asd_module.fft2_shift = lambda x, norm="ortho": FFT2Shift.apply(x)
    asd_module.ifft2_shift = (
        lambda magnitude, phase, norm="ortho":
        IFFT2Shift.apply(magnitude, phase))

    def dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape):
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape[:2])
        x1 = (x1 / max_shape[1]).clamp(0, 1) * max_shape[1]
        y1 = (y1 / max_shape[0]).clamp(0, 1) * max_shape[0]
        x2 = (x2 / max_shape[1]).clamp(0, 1) * max_shape[1]
        y2 = (y2 / max_shape[0]).clamp(0, 1) * max_shape[0]
        return x1, y1, x2, y2

    core_module = types.ModuleType("mmdet.core")
    export_module = types.ModuleType("mmdet.core.export")
    export_module.dynamic_clip_for_onnx = dynamic_clip_for_onnx
    core_module.export = export_module
    sys.modules["mmdet.core"] = core_module
    sys.modules["mmdet.core.export"] = export_module

    def nms_symbolic(g, boxes, scores, iou_threshold, offset,
                     score_threshold, max_num):
        iou_threshold = symbolic_helper._parse_arg(iou_threshold, "f")
        score_threshold = symbolic_helper._parse_arg(score_threshold, "f")
        max_num = symbolic_helper._parse_arg(max_num, "i")
        axes = g.op(
            "Constant", value_t=torch.tensor([0], dtype=torch.long))
        boxes = g.op("Unsqueeze", boxes, axes)
        scores = g.op("Unsqueeze", scores, axes)
        scores = g.op("Unsqueeze", scores, axes)
        max_output = g.op(
            "Constant",
            value_t=torch.tensor(
                [max_num if max_num > 0 else 2147483647], dtype=torch.long))
        iou = g.op(
            "Constant",
            value_t=torch.tensor([iou_threshold], dtype=torch.float))
        score = g.op(
            "Constant",
            value_t=torch.tensor([score_threshold], dtype=torch.float))
        selected = g.op(
            "NonMaxSuppression", boxes, scores, max_output, iou, score)
        column = g.op(
            "Constant", value_t=torch.tensor([2], dtype=torch.long))
        selected = g.op("Gather", selected, column, axis_i=1)
        squeeze_axes = g.op(
            "Constant", value_t=torch.tensor([1], dtype=torch.long))
        return g.op("Squeeze", selected, squeeze_axes)

    nms_module.NMSop.symbolic = staticmethod(nms_symbolic)

    class ExportNMS(torch.autograd.Function):
        @staticmethod
        def forward(ctx, boxes, scores, iou_threshold, score_threshold,
                    max_num):
            _, keep = original_nms(
                boxes,
                scores,
                float(iou_threshold),
                score_threshold=float(score_threshold),
                max_num=int(max_num))
            return keep

        @staticmethod
        def symbolic(g, boxes, scores, iou_threshold, score_threshold,
                     max_num):
            return nms_symbolic(
                g, boxes, scores, iou_threshold, 0, score_threshold, max_num)

    def export_multiclass_nms(multi_bboxes, multi_scores, score_thr,
                              nms_cfg, max_num=-1, score_factors=None,
                              return_inds=False, box_dim=4):
        num_classes = multi_scores.size(1) - 1
        if multi_bboxes.shape[1] > box_dim:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, box_dim)
        else:
            bboxes = multi_bboxes[:, None].expand(
                multi_scores.size(0), num_classes, box_dim)
        scores = multi_scores[:, :-1]
        labels = torch.arange(
            num_classes, dtype=torch.long, device=scores.device)
        labels = labels.view(1, -1).expand_as(scores)
        bboxes = bboxes.reshape(-1, box_dim)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        if score_factors is not None:
            factors = score_factors.view(-1, 1).expand(
                multi_scores.size(0), num_classes).reshape(-1)
            scores = scores * factors

        max_coordinate = bboxes.max()
        offsets = labels.to(bboxes) * (max_coordinate + 1)
        boxes_for_nms = bboxes + offsets[:, None]
        iou_threshold = nms_cfg.get(
            "iou_threshold", nms_cfg.get("iou_thr"))
        keep = ExportNMS.apply(
            boxes_for_nms, scores, float(iou_threshold), float(score_thr),
            int(max_num))
        detections = torch.cat([bboxes[keep], scores[keep, None]], dim=1)
        if return_inds:
            return detections, labels[keep], keep
        return detections, labels[keep]

    bbox_head_module.multiclass_nms = export_multiclass_nms

    def roi_align_symbolic(g, x, rois, output_size, spatial_scale,
                           sampling_ratio, pool_mode, aligned):
        from torch.onnx import TensorProtoDataType
        batch_indices = g.op(
            "Gather", rois,
            g.op(
                "Constant", value_t=torch.tensor([0], dtype=torch.long)),
            axis_i=1)
        axes = g.op(
            "Constant", value_t=torch.tensor([1], dtype=torch.long))
        batch_indices = g.op("Squeeze", batch_indices, axes)
        batch_indices = g.op(
            "Cast", batch_indices, to_i=TensorProtoDataType.INT64)
        rois = g.op(
            "Gather", rois,
            g.op(
                "Constant",
                value_t=torch.tensor([1, 2, 3, 4], dtype=torch.long)),
            axis_i=1)
        if aligned:
            offset = g.op(
                "Constant",
                value_t=torch.tensor(
                    [0.5 / spatial_scale], dtype=torch.float32))
            rois = g.op("Sub", rois, offset)
        return g.op(
            "RoiAlign", x, rois, batch_indices,
            output_height_i=output_size[0],
            output_width_i=output_size[1],
            spatial_scale_f=spatial_scale,
            sampling_ratio_i=max(0, sampling_ratio),
            mode_s=pool_mode)

    roi_align_module.RoIAlignFunction.symbolic = staticmethod(
        roi_align_symbolic)


class EndToEndModel(torch.nn.Module):
    def __init__(self, detector, data_sample):
        super().__init__()
        self.detector = detector
        self.data_sample = data_sample

    def forward(self, image):
        instances = self.detector.predict(
            image, [self.data_sample], rescale=False)[0].pred_instances
        return instances.bboxes, instances.scores, instances.labels.long()


def prepare_input(detector, cfg, image_path):
    pipeline = Compose(get_test_pipeline_cfg(cfg))
    item = pipeline(dict(img_path=str(image_path), img_id=0))
    data = detector.data_preprocessor(
        dict(inputs=[item["inputs"]], data_samples=[item["data_samples"]]),
        training=False)
    return data["inputs"], data["data_samples"][0]


def check_onnx(output_file):
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError(
            "ONNX validation requires `pip install onnx`.") from error

    graph = onnx.load(str(output_file))
    onnx.checker.check_model(graph)
    custom_counts = Counter(
        node.op_type for node in graph.graph.node if node.domain == "sfdnet")
    missing = REQUIRED_CUSTOM_OPS.difference(custom_counts)
    if missing:
        raise RuntimeError(
            "Exported graph is missing required custom ops: "
            + ", ".join(sorted(missing)))
    return custom_counts, len(graph.graph.node)


def main():
    args = parse_args()
    if not args.device.startswith("cuda"):
        raise ValueError("SFDNet export requires a CUDA device")

    image_path = validate_file(args.image, "image")
    checkpoint_path = validate_file(args.checkpoint, "checkpoint")
    config_path = validate_file(args.config, "config")
    output_file = (
        Path(args.output_file).expanduser() if args.output_file else
        Path("outputs") / f"{checkpoint_path.stem}.onnx")
    output_file = output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(config_path))
    score_thr, nms_thr = configure_model(cfg, args.score_thr, args.nms_thr)
    cfg.test_dataloader.dataset.pipeline = remove_annotation_loader(
        cfg.test_dataloader.dataset.pipeline)

    register_export_ops()
    detector = init_detector(
        cfg, str(checkpoint_path), device=args.device, palette="none")
    input_tensor, data_sample = prepare_input(detector, cfg, image_path)
    wrapper = EndToEndModel(detector, data_sample).eval()

    print(f"Input shape: {tuple(input_tensor.shape)}")
    print(f"Thresholds: score={score_thr}, nms={nms_thr}")
    with torch.no_grad():
        boxes, scores, labels = wrapper(input_tensor)
    print(f"PyTorch detections: {boxes.shape[0]}")

    torch.onnx.export(
        wrapper,
        input_tensor,
        str(output_file),
        opset_version=OPSET_VERSION,
        input_names=["input"],
        output_names=["boxes", "scores", "labels"],
        do_constant_folding=False,
        autograd_inlining=False)

    if not args.skip_check:
        custom_counts, node_count = check_onnx(output_file)
        summary = ", ".join(
            f"{name}={count}" for name, count in sorted(custom_counts.items()))
        print(f"ONNX checker: PASS ({node_count} nodes; {summary})")
    print(f"ONNX: {output_file}")
    print(
        "Runtime note: implement the sfdnet custom-domain operators before "
        "running this graph in ONNX Runtime or TensorRT.")


if __name__ == "__main__":
    main()
