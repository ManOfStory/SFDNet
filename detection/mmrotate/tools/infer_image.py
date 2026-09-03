#!/usr/bin/env python3
"""Run SFDNet OBB inference on one image and save image/JSON results."""

import argparse
import json
import sys
from pathlib import Path


MMROTATE_DIR = Path(__file__).resolve().parent.parent
DETECTION_DIR = MMROTATE_DIR.parent
if str(DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTION_DIR))

from mmcv import Config
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.datasets import DATASETS

import mmrotate  # noqa: F401,E402 - registers rotated detection components
import model  # noqa: F401,E402 - registers SFDNet model components


DEFAULT_CONFIG = MMROTATE_DIR / "configs/SFDNet/sodaa/SFDNet_Mamba.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SFDNet OBB inference on a single image")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("checkpoint", help="Trained checkpoint path")
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), help="Model config path")
    parser.add_argument(
        "--out-file",
        default=None,
        help="Visualization path (default: outputs/<image>_prediction.jpg)")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Detection JSON path (default: beside the visualization)")
    parser.add_argument(
        "--device", default="cuda:0", help="Inference device, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--score-thr", type=float, default=0.3, help="Score threshold")
    parser.add_argument(
        "--nms-thr",
        type=float,
        default=None,
        help="Final detection NMS IoU threshold (default: use config)")
    parser.add_argument(
        "--palette",
        default="dota",
        choices=("dota", "sar", "hrsc", "hrsc_classwise", "random"),
        help="Visualization palette")
    return parser.parse_args()


def set_classes_from_config(detector):
    """Use configured class names even if checkpoint metadata is incomplete."""
    dataset_cfg = detector.cfg.data.test
    while "dataset" in dataset_cfg:
        dataset_cfg = dataset_cfg.dataset
    dataset_cls = DATASETS.get(dataset_cfg.type)
    if dataset_cls is not None and hasattr(dataset_cls, "CLASSES"):
        detector.CLASSES = dataset_cls.CLASSES


def configure_test_thresholds(cfg, score_thr, nms_thr):
    """Set final detection thresholds and return the effective NMS value."""
    if not 0.0 <= score_thr <= 1.0:
        raise ValueError("--score-thr must be between 0 and 1")
    if nms_thr is not None and not 0.0 <= nms_thr <= 1.0:
        raise ValueError("--nms-thr must be between 0 and 1")

    rcnn_cfg = cfg.model.test_cfg.rcnn
    rcnn_cfg.score_thr = score_thr
    nms_cfg = rcnn_cfg.nms
    nms_key = "iou_thr" if "iou_thr" in nms_cfg else "iou_threshold"
    if nms_thr is not None:
        nms_cfg[nms_key] = nms_thr
    return float(nms_cfg[nms_key])


def serialize_result(result, classes, score_thr):
    detections = []
    for label, class_detections in enumerate(result):
        class_name = classes[label] if label < len(classes) else str(label)
        for detection in class_detections:
            score = float(detection[-1])
            if score < score_thr:
                continue
            detections.append({
                "class_id": label,
                "class_name": class_name,
                "score": round(score, 6),
                "obb_cxcywha": [
                    round(float(value), 3) for value in detection[:5]
                ],
            })
    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections


def main():
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    for path, label in (
            (image_path, "image"), (checkpoint_path, "checkpoint"),
            (config_path, "config")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    out_file = (Path(args.out_file).expanduser() if args.out_file else
                Path("outputs") / f"{image_path.stem}_prediction.jpg")
    out_file = out_file.resolve()
    json_out = (Path(args.json_out).expanduser() if args.json_out else
                out_file.with_suffix(".json"))
    json_out = json_out.resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(config_path))
    # A full detector checkpoint is loaded below; backbone pretraining is only
    # needed when training from scratch and may point to another machine.
    if "pretrained" in cfg.model.backbone:
        cfg.model.backbone.pretrained = None
    if "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None
    effective_nms_thr = configure_test_thresholds(
        cfg, args.score_thr, args.nms_thr)
    detector = init_detector(cfg, str(checkpoint_path), device=args.device)
    set_classes_from_config(detector)
    result = inference_detector(detector, str(image_path))

    show_result_pyplot(
        detector,
        str(image_path),
        result,
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=str(out_file))

    classes = tuple(detector.CLASSES)
    detections = serialize_result(result, classes, args.score_thr)
    payload = {
        "image": str(image_path),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "score_threshold": args.score_thr,
        "nms_iou_threshold": effective_nms_thr,
        "num_detections": len(detections),
        "detections": detections,
    }
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Detections: {len(detections)}")
    print(f"Thresholds: score={args.score_thr}, nms={effective_nms_thr}")
    print(f"Visualization: {out_file}")
    print(f"JSON: {json_out}")


if __name__ == "__main__":
    main()
