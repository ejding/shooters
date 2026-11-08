# RangeLab Pose Pipeline

An experimental data pipeline for learning how basketball shooting mechanics change
between made and missed long-range shots.

This repository begins with **pose extraction**, not a coaching model. It converts
each labelled basketball clip into a release-aligned sequence of body landmarks that
can later be used for interpretable analysis and temporal modelling.

## MVP scope

1. Use `Made Shot` and `Missed Shot` clips from BasketEvent as bootstrap data.
2. Crop to the labelled shooter when a per-frame bounding-box file is available.
3. Extract 33 MediaPipe body landmarks from every frame.
4. Save the landmarks, visibility scores, and source outcome label as JSONL.
5. Review an overlay video before trusting any downstream data.

BasketEvent is useful for proving the pipeline, but its NBA broadcast view is not the
target environment for a consumer shooting coach. A fixed-camera dataset of solo
three-point shooters is required before making player-facing coaching claims.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Input manifest

Create a CSV with `clip_id`, `video_path`, and `label`. Labels are `made` or `miss`.
`bbox_path` is optional. When supplied it should point to a JSON object whose keys are
zero-based frame numbers and whose values are `[x, y, width, height]` shooter boxes.

```csv
clip_id,video_path,label,bbox_path
game001_shot001,data/raw/game001.mp4,made,data/raw/game001_boxes.json
game001_shot002,data/raw/game002.mp4,miss,
```

## Extract pose sequences

```bash
python scripts/extract_pose.py \
  --manifest data/manifest.csv \
  --output data/processed/basketevent_pose.jsonl \
  --overlay-dir outputs/overlays
```

The overlay videos are a required quality check: inspect them before training. Low
visibility landmarks, wrong-player detections, and partial bodies should be filtered
out rather than treated as ground truth.

## Next experiments

- Select only BasketEvent `Made Shot` and `Missed Shot` clips, preferably clips with
  timing annotations.
- Locate release using the dataset's ball track and the shooting-wrist trajectory.
- Normalize each skeleton around the hip midpoint and torso scale.
- Start with interpretable features (knee bend, elbow angle, leg-to-arm timing), then
  compare made vs. missed shots per player.

