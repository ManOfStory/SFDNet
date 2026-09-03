#!/usr/bin/env python3
"""Run SFDNet HBB inference on one image and save image/JSON results."""

import argparse
import copy
import json
import sys
from pathlib import Path


DETECTION_DIR = Path(__file__).resolve().parent.parent
if str(DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTION_DIR))

import mmcv
from mmengine.config import Config

import model  # noqa: F401,E402 - registers SFDNet model components
from mmdet.apis import inference_detector, init_detector  # noqa: E402
from mmdet.registry import DATASETS  # noqa: E402
from mmdet.visualization import DetLocalVisualizer  # noqa: E402


DEFAULT_CONFIG = DETECTION_DIR / "configs/SFDNet/aitod/SFDNet_Mamba.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SFDNet HBB inference on a single image")
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
    return parser.parse_args()


def remove_annotation_loader(pipeline):
    """A standalone image has no annotation, so remove test-only GT loaders."""
    return [
        step for step in pipeline
        if step.get("type") not in ("LoadAnnotations", "mmdet.LoadAnnotations")
    ]


def set_dataset_meta_from_config(detector):
    """Use configured class names even if checkpoint metadata is incomplete."""
    dataset_cfg = detector.cfg.test_dataloader.dataset
    while "dataset" in dataset_cfg:
        dataset_cfg = dataset_cfg.dataset
    dataset_cls = DATASETS.get(dataset_cfg.type)
    if dataset_cls is not None and hasattr(dataset_cls, "METAINFO"):
        detector.dataset_meta.update(copy.deepcopy(dataset_cls.METAINFO))


def configure_test_thresholds(cfg, score_thr, nms_thr):
    """Set final detection thresholds and return the effective NMS value."""
    if not 0.0 <= score_thr <= 1.0:
        raise ValueError("--score-thr must be between 0 and 1")
    if nms_thr is not None and not 0.0 <= nms_thr <= 1.0:
        raise ValueError("--nms-thr must be between 0 and 1")

    rcnn_cfg = cfg.model.test_cfg.rcnn
    rcnn_cfg.score_thr = score_thr
    if nms_thr is not None:
        rcnn_cfg.nms.iou_threshold = nms_thr
    return float(rcnn_cfg.nms.iou_threshold)


def serialize_result(result, classes, score_thr):
    instances = result.pred_instances
    bboxes = instances.bboxes
    if hasattr(bboxes, "tensor"):
        bboxes = bboxes.tensor
    bboxes = bboxes.detach().cpu().tolist()
    scores = instances.scores.detach().cpu().tolist()
    labels = instances.labels.detach().cpu().tolist()

    detections = []
    for bbox, score, label in zip(bboxes, scores, labels):
        if score < score_thr:
            continue
        class_name = classes[label] if label < len(classes) else str(label)
        detections.append({
            "class_id": int(label),
            "class_name": class_name,
            "score": round(float(score), 6),
            "bbox_xyxy": [round(float(value), 3) for value in bbox],
        })
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
    cfg.test_dataloader.dataset.pipeline = remove_annotation_loader(
        cfg.test_dataloader.dataset.pipeline)
    detector = init_detector(
        cfg, str(checkpoint_path), device=args.device, palette="none")
    set_dataset_meta_from_config(detector)
    result = inference_detector(detector, str(image_path))

    visualizer = DetLocalVisualizer(name="sfdnet_single_image")
    visualizer.dataset_meta = detector.dataset_meta
    rgb_image = mmcv.imread(str(image_path), channel_order="rgb")
    visualizer.add_datasample(
        image_path.name,
        rgb_image,
        data_sample=result,
        draw_gt=False,
        show=False,
        out_file=str(out_file),
        pred_score_thr=args.score_thr)

    classes = tuple(detector.dataset_meta.get("classes", ()))
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
