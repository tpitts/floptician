# Scripts

Utility scripts for the floptician project — playing card detection for poker streams. Covers everything from extracting card images off scanned sheets to generating synthetic training data and running live detection overlays.

## YOLO Training Workflow

End-to-end steps to train a new card detection model:

### 1. Extract card images

Use the SAM-based scripts to segment scanned card sheets into individual card PNGs:

```bash
python scripts/sam_folder.py
```

Or use `sam_extract.py` / `fast_sam_extract.py` for alternative extraction methods.

### 2. Straighten cards

Auto-rotate and crop the extracted card PNGs so they're upright:

```bash
python scripts/straighten.py
```

### 3. Identify cards

Label and rename card images using OpenAI Vision:

```bash
python scripts/identify_cards.py
```

### 4. Generate training data

Produce a synthetic dataset with YOLO annotations:

```bash
python scripts/make_yolo_noise_2026.py
```

### 5. Train

```bash
yolo task=detect mode=train model=yolov8l.pt data="<dataset>/data.yaml" \
  epochs=80 imgsz=1280 degrees=11 translate=0.015 shear=0.035 \
  perspective=0.0003 flipud=0 fliplr=0 mixup=0 copy_paste=0 \
  patience=20 mosaic=0 scale=0.3 batch=-1 hsv_h=0.045
```

Key settings:
- `imgsz=1280` — high resolution for small card details
- `batch=-1` — auto batch size (fills available GPU memory)
- `patience=20` — early stopping if no improvement for 20 epochs
- `flipud=0 fliplr=0 mixup=0 copy_paste=0 mosaic=0` — augmentations disabled because the synthetic data already has augmentation baked in

### 6. Evaluate

Visualize false positives and negatives against ground truth:

```bash
python scripts/bad_detections.py
```

### 7. Export

Export the current model to a Core ML package with embedded non-maximum suppression,
then compare it with PyTorch across every labeled screenshot:

```bash
uv run --no-sync python scripts/convert.py
uv run --no-sync python scripts/verify_coreml.py
uv run --no-sync python scripts/verify_coreml.py --compute-unit cpu-only
uv run --no-sync python scripts/verify_coreml.py --compute-unit cpu-and-gpu
```

## Active Scripts

### Capture & Overlay

| Script | Description |
|--------|-------------|
| `virtualcam.py` | Webcam feed with live YOLO bounding box display |

### Training Data Generation

| Script | Description |
|--------|-------------|
| `make_yolo_noise_2026.py` | Generates synthetic training images with noise, blur, glare, and YOLO labels |
| `rotate_training.py` | Rotates training images 180° and adjusts bounding box annotations |

### Segmentation & Extraction

| Script | Description |
|--------|-------------|
| `fetch_gkards_deck.py` | Downloads a deck's card scans from gkards.com and converts them to transparent PNGs in `resources/decks/` (e.g. `python scripts/fetch_gkards_deck.py 000454 kem-deck-1`) |
| `sam_folder.py` | SAM2-based mask extraction from scanned card sheets |
| `sam_extract.py` | SAM2 extraction with contour-based processing |
| `fast_sam_extract.py` | FastSAM-based extraction (faster, less precise) |
| `extract.py` | Simple contour-based card extraction (no SAM) |
| `straighten.py` | Auto-rotates and crops transparent card PNGs to be upright |

### Analysis & Export

| Script | Description |
|--------|-------------|
| `bad_detections.py` | Visualizes false positives/negatives against ground truth |
| `convert.py` | Exports and validates the current YOLO model as a Core ML package |
| `verify_coreml.py` | Checks Core ML/PyTorch parity and reports inference latency |
| `identify_cards.py` | Uses OpenAI Vision API to identify and rename card images |

### Utilities

| Script | Description |
|--------|-------------|
| `_common.py` | Shared path helpers (`repo_path`, `resources_path`, etc.) |
