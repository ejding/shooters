#!/usr/bin/env python3
"""Extract MediaPipe pose landmarks from labelled basketball video clips.

The script is deliberately conservative: it records landmark confidence and allows
an optional shooter bounding box per frame. Downstream code should exclude low-quality
frames rather than silently treating guessed body positions as accurate mechanics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--model-complexity", choices=("0", "1", "2"), default="1")
    return parser.parse_args()


def load_boxes(path: str | None) -> dict[str, list[float]]:
    if not path:
        return {}
    with Path(path).open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object keyed by frame number: {path}")
    return payload


def crop_to_box(frame: Any, box: list[float] | None) -> tuple[Any, tuple[int, int]]:
    if box is None:
        return frame, (0, 0)
    x, y, width, height = (int(value) for value in box)
    frame_height, frame_width = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame_width, x + width), min(frame_height, y + height)
    if x1 <= x0 or y1 <= y0:
        return frame, (0, 0)
    return frame[y0:y1, x0:x1], (x0, y0)


def serialise_landmarks(result: Any, origin: tuple[int, int], crop_shape: tuple[int, int]) -> list[dict[str, float]]:
    crop_height, crop_width = crop_shape
    ox, oy = origin
    landmarks = []
    for index, landmark in enumerate(result.pose_landmarks.landmark):
        # x/y are saved in original-frame pixels; z remains MediaPipe's relative depth.
        landmarks.append(
            {
                "id": index,
                "x": round(ox + landmark.x * crop_width, 3),
                "y": round(oy + landmark.y * crop_height, 3),
                "z": round(landmark.z, 5),
                "visibility": round(landmark.visibility, 4),
            }
        )
    return landmarks


def overlay_pose(frame: Any, result: Any, origin: tuple[int, int], crop_shape: tuple[int, int]) -> Any:
    if not result.pose_landmarks:
        return frame
    ox, oy = origin
    crop_height, crop_width = crop_shape
    drawing = mp.solutions.drawing_utils
    connections = mp.solutions.pose.POSE_CONNECTIONS
    # Draw manually in original-frame coordinates because pose inference can use a crop.
    points = result.pose_landmarks.landmark
    for start, end in connections:
        a, b = points[start], points[end]
        p1 = (int(ox + a.x * crop_width), int(oy + a.y * crop_height))
        p2 = (int(ox + b.x * crop_width), int(oy + b.y * crop_height))
        cv2.line(frame, p1, p2, (0, 255, 0), 2)
    for point in points:
        p = (int(ox + point.x * crop_width), int(oy + point.y * crop_height))
        cv2.circle(frame, p, 3, (0, 128, 255), -1)
    return frame


def process_clip(row: dict[str, str], pose: Any, writer: Any, overlay_dir: Path | None) -> None:
    clip_id, label, video_path = row["clip_id"], row["label"].lower(), Path(row["video_path"])
    if label not in {"made", "miss"}:
        raise ValueError(f"{clip_id}: label must be 'made' or 'miss', got {label!r}")
    boxes = load_boxes(row.get("bbox_path"))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    overlay_writer = None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"{clip_id}.mp4"
        overlay_writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        crop, origin = crop_to_box(frame, boxes.get(str(frame_index)))
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_crop)
        record: dict[str, Any] = {
            "clip_id": clip_id,
            "label": label,
            "video_path": str(video_path),
            "frame": frame_index,
            "timestamp_s": round(frame_index / fps, 4),
            "frame_size": [frame_width, frame_height],
            "landmarks": None,
        }
        if result.pose_landmarks:
            record["landmarks"] = serialise_landmarks(result, origin, crop.shape[:2])
        writer.write(json.dumps(record) + "\n")
        if overlay_writer:
            overlay_writer.write(overlay_pose(frame, result, origin, crop.shape[:2]))
        frame_index += 1

    capture.release()
    if overlay_writer:
        overlay_writer.release()
    print(f"{clip_id}: wrote {frame_index} frames")


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(newline="") as manifest, args.output.open("w") as output:
        rows = list(csv.DictReader(manifest))
        required = {"clip_id", "video_path", "label"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"Manifest must include: {', '.join(sorted(required))}")
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=int(args.model_complexity),
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            for row in rows:
                process_clip(row, pose, output, args.overlay_dir)


if __name__ == "__main__":
    main()
