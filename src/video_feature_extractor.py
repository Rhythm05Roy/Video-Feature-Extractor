import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract


def ensure_video_path(video_path: Path) -> Path:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    return video_path


def read_frame_count(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to open video: {video_path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def detect_hard_cuts(
    video_path: Path, diff_threshold: float = 30.0, min_gap_frames: int = 5, frame_step: int = 1
) -> Dict[str, float]:
    """
    Counts hard cuts by measuring mean pixel differences between consecutive frames.
    A cut is counted when the mean difference exceeds the threshold and is spaced by min_gap_frames.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to open video: {video_path}")

    cuts = 0
    frame_idx = 0
    last_cut_frame = -min_gap_frames
    prev_gray = None

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            frame_diff = cv2.absdiff(gray, prev_gray)
            mean_diff = float(np.mean(frame_diff))
            if mean_diff > diff_threshold and (frame_idx - last_cut_frame) >= min_gap_frames:
                cuts += 1
                last_cut_frame = frame_idx
        prev_gray = gray
        frame_idx += 1

        if frame_step > 1:
            for _ in range(frame_step - 1):
                capture.grab()
                frame_idx += 1

    capture.release()

    return {
        "shot_cut_count": cuts,
        "frame_step_used": frame_step,
        "mean_diff_threshold": diff_threshold,
    }


def motion_analysis(
    video_path: Path,
    frame_step: int = 2,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
) -> Dict[str, float]:
    """
    Estimates average motion using Farneback optical flow across sampled frames.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to open video: {video_path}")

    prev_gray = None
    magnitudes: List[float] = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                pyr_scale,
                levels,
                winsize,
                iterations,
                poly_n,
                poly_sigma,
                0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(float(np.mean(mag)))

        prev_gray = gray

        if frame_step > 1:
            for _ in range(frame_step - 1):
                capture.grab()

    capture.release()

    avg_motion = float(np.mean(magnitudes)) if magnitudes else 0.0

    return {
        "average_motion_magnitude": avg_motion,
        "motion_samples": len(magnitudes),
        "frame_step_used": frame_step,
    }


def _extract_text_from_frame(frame: np.ndarray, min_confidence: float) -> Tuple[bool, List[str]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError:
        raise

    has_text = False
    keywords: List[str] = []
    for word, conf in zip(data.get("text", []), data.get("conf", [])):
        if not word or word.isspace():
            continue
        try:
            conf_val = float(conf)
        except (ValueError, TypeError):
            continue
        if conf_val >= min_confidence:
            has_text = True
            keywords.append(word.strip())
    return has_text, keywords


def text_detection(
    video_path: Path, frame_step: int = 15, min_confidence: float = 70.0
) -> Dict[str, object]:
    """
    Estimates how frequently text appears using pytesseract OCR on sampled frames.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to open video: {video_path}")

    total_frames = 0
    frames_with_text = 0
    keywords: Counter = Counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            total_frames += 1

            try:
                has_text, words = _extract_text_from_frame(frame, min_confidence)
            except pytesseract.TesseractNotFoundError:
                capture.release()
                return {
                    "text_present_ratio": 0.0,
                    "frames_with_text": 0,
                    "total_frames_evaluated": total_frames,
                    "keywords_top5": [],
                    "error": "Tesseract OCR binary not found on PATH. Install it and retry.",
                }

            if has_text:
                frames_with_text += 1
                keywords.update([word.lower() for word in words])

            if frame_step > 1:
                for _ in range(frame_step - 1):
                    capture.grab()
    finally:
        capture.release()

    ratio = (frames_with_text / total_frames) if total_frames else 0.0
    most_common = [word for word, _ in keywords.most_common(5)]

    return {
        "text_present_ratio": ratio,
        "frames_with_text": frames_with_text,
        "total_frames_evaluated": total_frames,
        "keywords_top5": most_common,
        "frame_step_used": frame_step,
        "min_confidence": min_confidence,
    }


def _load_class_names(names_path: Optional[Path]) -> List[str]:
    fallback = [
        "person",
        "bicycle",
        "car",
        "motorbike",
        "aeroplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "sofa",
        "pottedplant",
        "bed",
        "diningtable",
        "toilet",
        "tvmonitor",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    if names_path and names_path.is_file():
        content = names_path.read_text().strip().splitlines()
        cleaned = [line.strip() for line in content if line.strip()]
        return cleaned or fallback
    return fallback


def _yolo_output_names(net: cv2.dnn_Net) -> List[str]:
    layers = net.getLayerNames()
    return [layers[i - 1] for i in net.getUnconnectedOutLayers().flatten()]


def object_person_dominance(
    video_path: Path,
    yolo_cfg: Path,
    yolo_weights: Path,
    class_names_path: Optional[Path],
    frame_step: int = 15,
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.4,
) -> Dict[str, object]:
    """
    Uses YOLO (OpenCV DNN) to estimate person vs non-person dominance across sampled frames.
    """
    if not yolo_cfg.is_file() or not yolo_weights.is_file():
        return {
            "error": (
                "YOLO config/weights not found. Download yolov3.cfg and yolov3.weights "
                "into models/ or pass --yolo-cfg / --yolo-weights."
            ),
            "persons_detected": 0,
            "objects_detected": 0,
            "frames_evaluated": 0,
        }

    class_names = _load_class_names(class_names_path)
    net = cv2.dnn.readNetFromDarknet(str(yolo_cfg), str(yolo_weights))
    output_layers = _yolo_output_names(net)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to open video: {video_path}")

    persons = 0
    objects = 0
    frames_processed = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames_processed += 1

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layers)

        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence > conf_threshold:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    boxes.append([x, y, width, height])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        if len(idxs) > 0:
            for i in idxs.flatten():
                name = class_names[class_ids[i]] if class_ids[i] < len(class_names) else "object"
                if name == "person":
                    persons += 1
                else:
                    objects += 1

        if frame_step > 1:
            for _ in range(frame_step - 1):
                capture.grab()
    capture.release()

    total = persons + objects
    person_ratio = persons / total if total else 0.0
    object_ratio = objects / total if total else 0.0

    dominant = "person" if persons > objects else "object" if objects > persons else "tie"
    return {
        "persons_detected": persons,
        "objects_detected": objects,
        "person_ratio": person_ratio,
        "object_ratio": object_ratio,
        "dominant_category": dominant,
        "frames_evaluated": frames_processed,
        "frame_step_used": frame_step,
        "confidence_threshold": conf_threshold,
        "nms_threshold": nms_threshold,
        "yolo_cfg": str(yolo_cfg),
        "yolo_weights": str(yolo_weights),
        "class_names_source": str(class_names_path) if class_names_path else "embedded_coco",
    }


def extract_features(
    video_path: Path,
    features: List[str],
    cut_frame_step: int = 1,
    motion_frame_step: int = 2,
    text_frame_step: int = 15,
    object_frame_step: int = 15,
    diff_threshold: float = 30.0,
    min_gap_frames: int = 5,
    text_min_confidence: float = 70.0,
    yolo_cfg: Path = Path("models/yolov3.cfg"),
    yolo_weights: Path = Path("models/yolov3.weights"),
    yolo_names: Optional[Path] = None,
    object_conf_threshold: float = 0.5,
    object_nms_threshold: float = 0.4,
) -> Dict[str, object]:
    video_path = ensure_video_path(video_path)
    available_features = {"cuts", "motion", "text", "objects"}
    requested = [f for f in features if f in available_features]
    if not requested:
        raise ValueError(f"No valid features requested. Choose from {sorted(available_features)}")

    results: Dict[str, object] = {"video_path": str(video_path), "requested_features": requested}

    if "cuts" in requested:
        results["shot_cut_detection"] = detect_hard_cuts(
            video_path, diff_threshold=diff_threshold, min_gap_frames=min_gap_frames, frame_step=cut_frame_step
        )
    if "motion" in requested:
        results["motion_analysis"] = motion_analysis(
            video_path, frame_step=motion_frame_step
        )
    if "text" in requested:
        results["text_detection"] = text_detection(
            video_path, frame_step=text_frame_step, min_confidence=text_min_confidence
        )
    if "objects" in requested:
        results["object_person_dominance"] = object_person_dominance(
            video_path=video_path,
            yolo_cfg=yolo_cfg,
            yolo_weights=yolo_weights,
            class_names_path=yolo_names,
            frame_step=object_frame_step,
            conf_threshold=object_conf_threshold,
            nms_threshold=object_nms_threshold,
        )

    results["frame_count"] = read_frame_count(video_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract visual features from a local video file.")
    parser.add_argument("video_path", type=Path, help="Path to the video file to analyze.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=["cuts", "motion", "text", "objects"],
        choices=["cuts", "motion", "text", "objects"],
        help="Which features to compute.",
    )
    parser.add_argument("--cut-frame-step", type=int, default=1, help="Frame step for shot cut detection.")
    parser.add_argument("--motion-frame-step", type=int, default=2, help="Frame step for optical flow.")
    parser.add_argument(
        "--text-frame-step", type=int, default=15, help="Frame step for OCR sampling to limit processing cost."
    )
    parser.add_argument(
        "--object-frame-step",
        type=int,
        default=15,
        help="Frame step for YOLO sampling to reduce compute cost.",
    )
    parser.add_argument(
        "--diff-threshold", type=float, default=30.0, help="Mean pixel difference threshold for hard cuts."
    )
    parser.add_argument(
        "--min-gap-frames", type=int, default=5, help="Minimum frame gap between detected hard cuts."
    )
    parser.add_argument(
        "--text-min-confidence",
        type=float,
        default=70.0,
        help="Minimum OCR confidence score to count a word.",
    )
    parser.add_argument(
        "--yolo-cfg",
        type=Path,
        default=Path("models/yolov3.cfg"),
        help="Path to YOLO config file.",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=Path("models/yolov3.weights"),
        help="Path to YOLO weights file.",
    )
    parser.add_argument(
        "--yolo-names",
        type=Path,
        help="Optional path to class names file (coco.names). Defaults to embedded list.",
    )
    parser.add_argument(
        "--object-conf-threshold",
        type=float,
        default=0.5,
        help="YOLO confidence threshold for detections.",
    )
    parser.add_argument(
        "--object-nms-threshold",
        type=float,
        default=0.4,
        help="YOLO non-max suppression threshold.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save JSON output. Defaults to stdout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    features = extract_features(
        video_path=args.video_path,
        features=args.features,
        cut_frame_step=args.cut_frame_step,
        motion_frame_step=args.motion_frame_step,
        text_frame_step=args.text_frame_step,
        object_frame_step=args.object_frame_step,
        diff_threshold=args.diff_threshold,
        min_gap_frames=args.min_gap_frames,
        text_min_confidence=args.text_min_confidence,
        yolo_cfg=args.yolo_cfg,
        yolo_weights=args.yolo_weights,
        yolo_names=args.yolo_names,
        object_conf_threshold=args.object_conf_threshold,
        object_nms_threshold=args.object_nms_threshold,
    )

    output_json = json.dumps(features, indent=2)
    if args.output:
        args.output.write_text(output_json)
        print(f"Wrote feature output to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
