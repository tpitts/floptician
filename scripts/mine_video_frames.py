"""Mine sharp, card-bearing, non-repetitive frames from poker videos.

Usage:
    python scripts/mine_video_frames.py resources/source-videos/4kU_Fkj6H0k.mp4
    python scripts/mine_video_frames.py --all
    python scripts/mine_video_frames.py <video> --start 600 --duration 300   # tuning slice

Two passes per video:
  A) Calibrate: ~150 random-seek samples -> sharpness baseline + burned-in
     graphics-overlay mask (card detections that recur at the same position).
  B) Mine: single sequential ffmpeg decode of KEYFRAMES ONLY (-skip_frame nokey),
     piped raw into this process. YouTube encodes a keyframe every ~2-4s and at
     scene cuts, which matches the sampling cadence we want while skipping ~95%
     of the decode work; timestamps come from the container index via ffprobe.
     A sample is kept when it is sharp, has >= MIN_CARDS real (non-overlay) card
     detections, and shows a new board state (card-class signature change) or a
     clearly new scene (perceptual-hash distance). Frames whose cards read
     sideways are rotated upright before saving.

Output: resources/mined-frames/<video-stem>/t<seconds>.png, manifest.csv,
and a contact-sheet montage for human review.
"""

import argparse
import csv
import json
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from _common import repo_path, resources_path

MODEL_PATH = repo_path("models", "yolov8l-2024-07-30.pt")
SAMPLE_INTERVAL_S = 2.0
CALIB_SAMPLES = 150
CONF_MIN = 0.5
CALIB_CONF = 0.35
MIN_CARDS = 2
MEAN_CONF_MIN = 0.70        # angled side-camera shots detect weakly; direct shots score ~0.9
UPRIGHT_ASPECT = (0.45, 0.95)
SIDEWAYS_ASPECT = (1.05, 2.2)
MAX_BOX_FRAC = 0.6          # reject a "card" filling most of the frame
OVERLAY_GRID_W = 48
OVERLAY_FRAC = 0.35         # cell is overlay if cards recur there in >35% of samples
BAND_EDGE_FRAC = 0.28       # top/bottom edge bands where burned-in card graphics live
BAND_TRIGGER = 0.30         # mask a band if cards recur there in >=30% of samples
EDGE_EXCLUDE_FRAC = 0.15    # always ignore dets this close to the encoded top/bottom edge:
                            # graphics strips live there; real overhead boards sit center-frame
SHARPNESS_FLOOR = 30.0
SHARPNESS_REL = 0.5         # candidate must reach 50% of the per-video baseline
DHASH_MIN = 6               # below this hamming distance to any recent keep -> duplicate
DHASH_NEW_SCENE = 12        # at/above this distance to all recent keeps -> new scene
DHASH_HISTORY = 12


def keyframe_times(path: Path) -> list[float]:
    """Keyframe timestamps from the container index — no decoding involved."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout
    times = []
    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) >= 2 and "K" in parts[1]:
            try:
                times.append(float(parts[0]))
            except ValueError:
                continue
    times.sort()
    return times


def ffprobe_info(path: Path) -> tuple[int, int, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True).stdout
    meta = json.loads(out)
    stream = next(s for s in meta["streams"] if s["codec_type"] == "video")
    return int(stream["width"]), int(stream["height"]), float(meta["format"]["duration"])


def sharpness(bgr: np.ndarray) -> float:
    h, w = bgr.shape[:2]
    scale = 960 / max(h, w)
    small = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dhash(bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(cv2.resize(bgr, (9, 8)), cv2.COLOR_BGR2GRAY)
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | int(gray[row, col] > gray[row, col + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def card_detections(model, bgr: np.ndarray, conf: float):
    """Run the detector and return card-plausible boxes: (cls, conf, x1, y1, x2, y2)."""
    r = model.predict(bgr, conf=conf, imgsz=1280, verbose=False)[0]
    h, w = bgr.shape[:2]
    dets = []
    for c, cf, box in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0 or bw > MAX_BOX_FRAC * w or bh > MAX_BOX_FRAC * h:
            continue
        aspect = bw / bh
        if UPRIGHT_ASPECT[0] <= aspect <= UPRIGHT_ASPECT[1] or SIDEWAYS_ASPECT[0] <= aspect <= SIDEWAYS_ASPECT[1]:
            dets.append((int(c), float(cf), x1, y1, x2, y2))
    return dets


class OverlayMask:
    """Grid of cells where card detections recur at fixed positions (burned-in graphics)."""

    def __init__(self, width: int, height: int):
        self.gw = OVERLAY_GRID_W
        self.gh = max(1, round(OVERLAY_GRID_W * height / width))
        self.width, self.height = width, height
        self.counts = np.zeros((self.gh, self.gw), dtype=np.int32)
        self.samples = 0

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        cx = min(self.gw - 1, int(x / self.width * self.gw))
        cy = min(self.gh - 1, int(y / self.height * self.gh))
        return cy, cx

    def accumulate(self, dets) -> None:
        self.samples += 1
        hit = np.zeros_like(self.counts, dtype=bool)
        for _, _, x1, y1, x2, y2 in dets:
            cy, cx = self._cell((x1 + x2) / 2, (y1 + y2) / 2)
            hit[cy, cx] = True
        self.counts += hit

    def finalize(self) -> None:
        frac = self.counts / max(1, self.samples)
        mask = (frac > OVERLAY_FRAC).astype(np.uint8)
        self.mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))

    def mask_band(self, y0: float, y1: float) -> None:
        """Mark every cell whose center lies in pixel range [y0, y1] as overlay."""
        for cy in range(self.gh):
            center = (cy + 0.5) / self.gh * self.height
            if y0 <= center <= y1:
                self.mask[cy, :] = 1

    def is_overlay(self, x: float, y: float) -> bool:
        cy, cx = self._cell(x, y)
        return bool(self.mask[cy, cx])

    def rotated90cw(self) -> "OverlayMask":
        m = OverlayMask(self.height, self.width)
        m.mask = np.rot90(self.mask, k=-1)
        m.gh, m.gw = m.mask.shape
        return m


def drop_edge_dets(dets, width: int, height: int, edges: tuple[str, ...]):
    out = []
    for d in dets:
        cx, cy = (d[2] + d[4]) / 2, (d[3] + d[5]) / 2
        if "top" in edges and cy < EDGE_EXCLUDE_FRAC * height:
            continue
        if "bottom" in edges and cy > (1 - EDGE_EXCLUDE_FRAC) * height:
            continue
        if "left" in edges and cx < EDGE_EXCLUDE_FRAC * width:
            continue
        if "right" in edges and cx > (1 - EDGE_EXCLUDE_FRAC) * width:
            continue
        out.append(d)
    return out


def real_cards(dets, overlay: OverlayMask):
    return [d for d in dets if not overlay.is_overlay((d[2] + d[4]) / 2, (d[3] + d[5]) / 2)]


def median_aspect(dets) -> float:
    if not dets:
        return 0.65
    return float(np.median([(x2 - x1) / (y2 - y1) for _, _, x1, y1, x2, y2 in dets]))


def calibrate(path: Path, model, width: int, height: int, duration: float):
    cap = cv2.VideoCapture(str(path))
    overlay = OverlayMask(width, height)
    sharps = []
    n_valid = top_hits = bottom_hits = 0
    top_y2s, bottom_y1s = [], []
    times = np.linspace(duration * 0.02, duration * 0.98, CALIB_SAMPLES)
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        n_valid += 1
        sharps.append(sharpness(frame))
        dets = card_detections(model, frame, CALIB_CONF)
        overlay.accumulate(dets)
        top = [d for d in dets if (d[3] + d[5]) / 2 < BAND_EDGE_FRAC * height]
        bottom = [d for d in dets if (d[3] + d[5]) / 2 > (1 - BAND_EDGE_FRAC) * height]
        if top:
            top_hits += 1
            top_y2s.append(max(d[5] for d in top))
        if bottom:
            bottom_hits += 1
            bottom_y1s.append(min(d[3] for d in bottom))
    cap.release()
    overlay.finalize()
    if n_valid and top_hits / n_valid >= BAND_TRIGGER:
        overlay.mask_band(0, min(0.4 * height, max(top_y2s) + 0.02 * height))
    if n_valid and bottom_hits / n_valid >= BAND_TRIGGER:
        overlay.mask_band(max(0.6 * height, min(bottom_y1s) - 0.02 * height), height)
    baseline = float(np.median(sharps)) if sharps else SHARPNESS_FLOOR
    return overlay, baseline


def mine(path: Path, out_root: Path, max_frames: int, start: float, duration_limit: float | None) -> None:
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))

    width, height, duration = ffprobe_info(path)
    if duration_limit:
        duration = min(duration - start, duration_limit)
    out_dir = out_root / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{path.stem}] {width}x{height}, {duration/60:.1f} min to scan; calibrating...")
    overlay, baseline = calibrate(path, model, width, height, ffprobe_info(path)[2])
    overlay_cw = overlay.rotated90cw()
    thresh = max(SHARPNESS_FLOOR, SHARPNESS_REL * baseline)
    print(f"[{path.stem}] sharpness baseline {baseline:.0f} (gate {thresh:.0f}), "
          f"overlay cells {int(overlay.mask.sum())}/{overlay.mask.size}")

    kf_times = [t for t in keyframe_times(path)
                if t >= start and (not duration_limit or t <= start + duration_limit)]
    print(f"[{path.stem}] {len(kf_times)} keyframes in scan range")

    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-skip_frame", "nokey"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if duration_limit:
        cmd += ["-t", str(duration_limit)]
    cmd += ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=width * height * 3 * 4)

    frame_bytes = width * height * 3
    manifest = []
    hashes: deque[int] = deque(maxlen=DHASH_HISTORY)
    last_sig: frozenset[int] = frozenset()
    idx = kept = 0
    last_t = -1e9
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            t = kf_times[idx] if idx < len(kf_times) else start + idx * 2.0
            idx += 1
            if t - last_t < SAMPLE_INTERVAL_S:  # keyframes can cluster at cuts
                continue
            last_t = t
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)

            if sharpness(frame) < thresh:
                continue
            dets = real_cards(card_detections(model, frame, CONF_MIN), overlay)
            dets = drop_edge_dets(dets, width, height, ("top", "bottom"))
            if len(dets) < MIN_CARDS:
                continue

            save_frame = frame
            if median_aspect(dets) >= SIDEWAYS_ASPECT[0]:  # cards read sideways -> normalize
                save_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                dets = real_cards(card_detections(model, save_frame, CONF_MIN), overlay_cw)
                # original top/bottom edges became the right/left edges after CW rotation
                dets = drop_edge_dets(dets, height, width, ("left", "right"))
                if len(dets) < MIN_CARDS:
                    continue

            if float(np.mean([d[1] for d in dets])) < MEAN_CONF_MIN:
                continue
            sig = frozenset(d[0] for d in dets)
            h = dhash(save_frame)
            dists = [hamming(h, old) for old in hashes]
            if dists and min(dists) < DHASH_MIN:
                continue
            if sig == last_sig and not (dists and min(dists) >= DHASH_NEW_SCENE):
                continue

            name = f"t{int(t):06d}.png"
            cv2.imwrite(str(out_dir / name), save_frame)
            manifest.append({
                "file": name, "t_seconds": int(t), "n_cards": len(dets),
                "classes": " ".join(model.names[d[0]] for d in sorted(dets, key=lambda d: d[2])),
                "rotated": save_frame is not frame,
                "mean_conf": round(float(np.mean([d[1] for d in dets])), 3),
            })
            hashes.append(h)
            last_sig = sig
            kept += 1
            if kept % 25 == 0:
                print(f"[{path.stem}] kept {kept} (t={int(t)}s)")
            if kept >= max_frames:
                print(f"[{path.stem}] hit cap of {max_frames}")
                break
    finally:
        proc.stdout.close()
        proc.terminate()

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "t_seconds", "n_cards", "classes", "rotated", "mean_conf"])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"[{path.stem}] done: {kept} frames from {idx} samples -> {out_dir}")
    if kept:
        subprocess.run(["magick", "montage", str(out_dir / "t*.png"), "-tile", "8x",
                        "-geometry", "240x+2+2", "-label", "%t",
                        str(out_dir / "contact-sheet.jpg")], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine card frames from poker videos.")
    parser.add_argument("videos", nargs="*", help="video file(s); or use --all")
    parser.add_argument("--all", action="store_true", help="process every mp4 in resources/source-videos")
    parser.add_argument("--out", type=Path, default=resources_path("mined-frames"))
    parser.add_argument("--max-frames", type=int, default=300, help="per-video keep cap")
    parser.add_argument("--start", type=float, default=0.0, help="seconds to skip from the beginning")
    parser.add_argument("--duration", type=float, default=None, help="only scan this many seconds (tuning)")
    args = parser.parse_args()

    videos = [Path(v) for v in args.videos]
    if args.all:
        videos += sorted(resources_path("source-videos").glob("*.mp4"))
    if not videos:
        parser.error("no videos given (pass paths or --all)")
    for v in videos:
        mine(v, args.out, args.max_frames, args.start, args.duration)


if __name__ == "__main__":
    main()
