#!/usr/bin/env python3
"""Export an SFDNet OBB checkpoint to a fixed-shape ONNX graph.

The Spatial-Mamba, ASD, rotated RoIAlign, and rotated NMS operations are
represented as explicit ``sfdnet`` custom-domain nodes. A deployment runtime
must provide implementations for those nodes; stock ONNX Runtime cannot execute
them.
"""

import argparse
import importlib
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.onnx import register_custom_op_symbolic, symbolic_helper


MMROTATE_DIR = Path(__file__).resolve().parent.parent
DETECTION_DIR = MMROTATE_DIR.parent
if str(DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTION_DIR))
if str(MMROTATE_DIR) not in sys.path:
    sys.path.insert(0, str(MMROTATE_DIR))

import model  # noqa: F401,E402 - registers SFDNet model components
import mmrotate  # noqa: F401,E402 - registers rotated detection components
from mmcv import Config  # noqa: E402
from mmcv.parallel import collate, scatter  # noqa: E402
from mmdet.apis import init_detector  # noqa: E402
from mmdet.datasets import replace_ImageToTensor  # noqa: E402
from mmdet.datasets.pipelines import Compose  # noqa: E402


DEFAULT_CONFIG = MMROTATE_DIR / "configs/SFDNet/sodaa/SFDNet_Mamba.py"
OPSET_VERSION = 13
REQUIRED_CUSTOM_OPS = {
    "SelectiveScan",
    "FFT2Shift",
    "IFFT2Shift",
    "RoIAlignRotated",
    "RotatedNMS",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export an SFDNet OBB checkpoint to ONNX")
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
        help="Final rotated-NMS IoU threshold (default: use config)")
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
        nms_key = "iou_thr" if "iou_thr" in test_cfg.nms else "iou_threshold"
        test_cfg.nms[nms_key] = nms_thr
    nms_key = "iou_thr" if "iou_thr" in test_cfg.nms else "iou_threshold"
    return float(test_cfg.score_thr), float(test_cfg.nms[nms_key])


def register_export_ops():
    """Register semantically equivalent ONNX mappings without editing models."""
    import models.spatialmamba as spatialmamba
    import models.utils as backbone_utils
    import mmrotate.models.necks.asd as asd_module
    import mmrotate.models.necks.utils.utils as neck_utils
    import mmrotate.models.roi_heads.bbox_heads.rotated_bbox_head as bbox_module

    nms_module = importlib.import_module("mmcv.ops.nms")
    roi_align_module = importlib.import_module("mmcv.ops.roi_align_rotated")
    rotated_nms_module = importlib.import_module(
        "mmrotate.core.post_processing.bbox_nms_rotated")
    original_nms_rotated = nms_module.nms_rotated

    def expm1_symbolic(g, x):
        one = g.op("Constant", value_t=torch.tensor(1.0))
        return g.op("Sub", g.op("Exp", x), one)

    register_custom_op_symbolic("aten::expm1", expm1_symbolic, OPSET_VERSION)

    def atan2_symbolic(g, y, x):
        zero = g.op("Constant", value_t=torch.tensor(0.0))
        pi = g.op("Constant", value_t=torch.tensor(torch.pi))
        half_pi = g.op("Constant", value_t=torch.tensor(torch.pi / 2))
        negative_pi = g.op("Neg", pi)
        negative_half_pi = g.op("Neg", half_pi)
        angle = g.op("Atan", g.op("Div", y, x))
        upper_half = g.op("GreaterOrEqual", y, zero)
        x_negative = g.op("Less", x, zero)
        angle = g.op(
            "Where", x_negative,
            g.op("Add", angle,
                 g.op("Where", upper_half, pi, negative_pi)),
            angle)
        y_positive = g.op("Greater", y, zero)
        y_negative = g.op("Less", y, zero)
        vertical = g.op(
            "Where", y_positive, half_pi,
            g.op("Where", y_negative, negative_half_pi, zero))
        return g.op("Where", g.op("Equal", x, zero), vertical, angle)

    register_custom_op_symbolic("aten::atan2", atan2_symbolic, OPSET_VERSION)

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

    def roi_align_rotated_symbolic(g, x, rois, output_size, spatial_scale,
                                   sampling_ratio, aligned, clockwise):
        if isinstance(output_size, int):
            out_h = out_w = output_size
        else:
            out_h, out_w = output_size
        output = g.op(
            "sfdnet::RoIAlignRotated", x, rois,
            output_height_i=out_h,
            output_width_i=out_w,
            spatial_scale_f=spatial_scale,
            sampling_ratio_i=sampling_ratio,
            aligned_i=aligned,
            clockwise_i=clockwise)
        x_sizes = x.type().sizes()
        if x_sizes:
            output.setType(x.type().with_sizes(
                [None, x_sizes[1], out_h, out_w]))
        return output

    roi_align_module.RoIAlignRotatedFunction.symbolic = staticmethod(
        roi_align_rotated_symbolic)

    class RotatedNMS(torch.autograd.Function):
        @staticmethod
        def forward(ctx, boxes, scores, iou_threshold, score_threshold,
                    max_num):
            valid = scores > score_threshold
            valid_indices = valid.nonzero(as_tuple=False).squeeze(1)
            boxes = boxes[valid_indices]
            scores = scores[valid_indices]
            if boxes.numel() == 0:
                return valid_indices
            _, keep = original_nms_rotated(boxes, scores, iou_threshold)
            if max_num > 0:
                keep = keep[:max_num]
            return valid_indices[keep]

        @staticmethod
        def symbolic(g, boxes, scores, iou_threshold, score_threshold,
                     max_num):
            iou_threshold = symbolic_helper._parse_arg(iou_threshold, "f")
            score_threshold = symbolic_helper._parse_arg(score_threshold, "f")
            max_num = symbolic_helper._parse_arg(max_num, "i")
            output = g.op(
                "sfdnet::RotatedNMS", boxes, scores,
                iou_threshold_f=iou_threshold,
                score_threshold_f=score_threshold,
                max_output_i=max_num)
            return output.setType(scores.type().with_dtype(torch.long))

    def export_multiclass_nms_rotated(multi_bboxes, multi_scores, score_thr,
                                      nms, max_num=-1, score_factors=None,
                                      return_inds=False):
        num_classes = multi_scores.size(1) - 1
        if multi_bboxes.shape[1] > 5:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
        else:
            bboxes = multi_bboxes[:, None].expand(
                multi_scores.size(0), num_classes, 5)
        scores = multi_scores[:, :-1]
        labels = torch.arange(
            num_classes, dtype=torch.long, device=scores.device)
        labels = labels.view(1, -1).expand_as(scores)
        bboxes = bboxes.reshape(-1, 5)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        if score_factors is not None:
            factors = score_factors.view(-1, 1).expand(
                multi_scores.size(0), num_classes).reshape(-1)
            scores = scores * factors

        max_coordinate = bboxes[:, :2].max() + bboxes[:, 2:4].max()
        offsets = labels.to(bboxes) * (max_coordinate + 1)
        boxes_for_nms = bboxes.clone()
        boxes_for_nms[:, :2] = boxes_for_nms[:, :2] + offsets[:, None]
        keep = RotatedNMS.apply(
            boxes_for_nms, scores, float(nms.iou_thr), float(score_thr),
            int(max_num))
        detections = torch.cat([bboxes[keep], scores[keep, None]], dim=1)
        if return_inds:
            return detections, labels[keep], keep
        return detections, labels[keep]

    rotated_nms_module.multiclass_nms_rotated = export_multiclass_nms_rotated
    bbox_module.multiclass_nms_rotated = export_multiclass_nms_rotated


class EndToEndModel(torch.nn.Module):
    def __init__(self, detector, img_metas):
        super().__init__()
        self.detector = detector
        self.img_metas = img_metas

    def forward(self, image):
        features = self.detector.extract_feat(image)
        proposals = self.detector.rpn_head.simple_test_rpn(
            features, self.img_metas)
        det_bboxes, det_labels = self.detector.roi_head.simple_test_bboxes(
            features,
            self.img_metas,
            proposals,
            self.detector.roi_head.test_cfg,
            rescale=False)
        detections = det_bboxes[0]
        return detections[:, :5], detections[:, 5], det_labels[0].long()


def prepare_input(detector, cfg, image_path, device):
    pipeline = Compose(replace_ImageToTensor(cfg.data.test.pipeline))
    data = pipeline(dict(
        img_info=dict(filename=str(image_path)), img_prefix=None))
    data = collate([data], samples_per_gpu=1)
    data["img_metas"] = [item.data[0] for item in data["img_metas"]]
    data["img"] = [item.data[0] for item in data["img"]]
    data = scatter(data, [device])[0]
    return data["img"][0], data["img_metas"][0]


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

    register_export_ops()
    detector = init_detector(cfg, str(checkpoint_path), device=args.device)
    input_tensor, img_metas = prepare_input(
        detector, cfg, image_path, args.device)
    wrapper = EndToEndModel(detector, img_metas).eval()

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
        do_constant_folding=False)

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
