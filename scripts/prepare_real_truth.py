"""Build the truth-v2 test set from LLM-labeled real frames (resources/mined-frames).

Each labeled source frame becomes up to 6 test images: median card width scaled
to each of the generator's size ladder {63, 91, 120}, times a 180-degree
rotation of each. Frames are cropped/padded onto a 1280x720 black canvas
(matching the live letterbox in yolo_processor.letterbox_image). A size variant
is skipped when the frame's card cluster cannot fit at that scale.

Output mirrors resources/truth-v1 layout (images/test + labels/test) so the
training-data generator can adopt it by pointing TRUTH_DIR at truth-v2, and a
data.yaml is written for standalone evaluation:

    yolo val model=<weights> data=resources/truth-v2/data.yaml split=test imgsz=1280
"""

import argparse
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from _common import resources_path

CANVAS_W, CANVAS_H = 1280, 720
TARGET_WIDTHS = [63, 91, 120]
MARGIN = 8  # keep the card cluster this many px inside the canvas

CARD_CLASSES = ['2c','2d','2h','2s','3c','3d','3h','3s','4c','4d','4h','4s','5c','5d','5h','5s',
                '6c','6d','6h','6s','7c','7d','7h','7s','8c','8d','8h','8s','9c','9d','9h','9s',
                'Ac','Ad','Ah','As','Jc','Jd','Jh','Js','Kc','Kd','Kh','Ks','Qc','Qd','Qh','Qs',
                'Tc','Td','Th','Ts']


def load_labels(path: Path, w: int, h: int):
    boxes = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            ci, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            boxes.append((ci, cx * w, cy * h, bw * w, bh * h))
    return boxes


def make_variant(img, boxes, target_w):
    """Scale so the median card width hits target_w, crop/pad to the canvas."""
    med = median(b[3] for b in boxes)
    s = target_w / med
    sw, sh = int(round(img.shape[1] * s)), int(round(img.shape[0] * s))
    sboxes = [(ci, cx * s, cy * s, bw * s, bh * s) for ci, cx, cy, bw, bh in boxes]

    x1 = min(cx - bw / 2 for _, cx, _, bw, _ in sboxes)
    x2 = max(cx + bw / 2 for _, cx, _, bw, _ in sboxes)
    y1 = min(cy - bh / 2 for _, _, cy, _, bh in sboxes)
    y2 = max(cy + bh / 2 for _, _, cy, _, bh in sboxes)
    if x2 - x1 > CANVAS_W - 2 * MARGIN or y2 - y1 > CANVAS_H - 2 * MARGIN:
        return None, None  # cluster does not fit at this scale

    scaled = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    cw, ch = min(sw, CANVAS_W), min(sh, CANVAS_H)
    cx0 = int(np.clip((x1 + x2) / 2 - cw / 2, 0, sw - cw))
    cy0 = int(np.clip((y1 + y2) / 2 - ch / 2, 0, sh - ch))
    # nudge the window so the cluster is fully inside it
    cx0 = int(np.clip(cx0, x2 + MARGIN - cw, x1 - MARGIN)) if cw < x2 - x1 + 2 * MARGIN else cx0
    crop = scaled[cy0:cy0 + ch, cx0:cx0 + cw]

    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    ox, oy = (CANVAS_W - cw) // 2, (CANVAS_H - ch) // 2
    canvas[oy:oy + ch, ox:ox + cw] = crop

    out = []
    for ci, cx, cy, bw, bh in sboxes:
        nx, ny = cx - cx0 + ox, cy - cy0 + oy
        if not (MARGIN / 2 <= nx - bw / 2 and nx + bw / 2 <= CANVAS_W - MARGIN / 2
                and MARGIN / 2 <= ny - bh / 2 and ny + bh / 2 <= CANVAS_H - MARGIN / 2):
            return None, None  # a card landed outside the window; treat variant as infeasible
        out.append((ci, nx / CANVAS_W, ny / CANVAS_H, bw / CANVAS_W, bh / CANVAS_H))
    return canvas, out


def rotate180(canvas, labels):
    return cv2.rotate(canvas, cv2.ROTATE_180), [(ci, 1 - cx, 1 - cy, bw, bh) for ci, cx, cy, bw, bh in labels]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build truth-v2 test set from mined frames.")
    parser.add_argument("--src", type=Path, default=resources_path("mined-frames"))
    parser.add_argument("--out", type=Path, default=resources_path("truth-v2"))
    args = parser.parse_args()

    img_dir = args.out / "images" / "test"
    lbl_dir = args.out / "labels" / "test"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    made = skipped = 0
    for vid_dir in sorted(d for d in args.src.iterdir() if d.is_dir() and not d.name.startswith("_") and d.name != "logs"):
        for frame in sorted(vid_dir.glob("t*.png")):
            lblf = vid_dir / "labels_llm" / (frame.stem + ".txt")
            if not lblf.exists():
                continue
            img = cv2.imread(str(frame))
            boxes = load_labels(lblf, img.shape[1], img.shape[0])
            if not boxes:
                continue
            for tw in TARGET_WIDTHS:
                canvas, labels = make_variant(img, boxes, tw)
                if canvas is None:
                    skipped += 1
                    continue
                for rot, (cnv, lbls) in (("r0", (canvas, labels)), ("r180", rotate180(canvas, labels))):
                    name = f"{vid_dir.name}_{frame.stem}_w{tw}_{rot}"
                    cv2.imwrite(str(img_dir / f"{name}.png"), cnv)
                    (lbl_dir / f"{name}.txt").write_text(
                        "\n".join(f"{ci} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for ci, cx, cy, bw, bh in lbls) + "\n")
                    made += 1
        print(f"{vid_dir.name}: done ({made} images so far)")

    names = {i: n for i, n in enumerate(CARD_CLASSES)}
    yaml_lines = [f"path: {args.out.resolve()}", "train: images/test", "val: images/test",
                  "test: images/test", f"nc: {len(CARD_CLASSES)}", "names:"]
    yaml_lines += [f"  {i}: {n}" for i, n in names.items()]
    (args.out / "data.yaml").write_text("\n".join(yaml_lines) + "\n")
    print(f"\n{made} test images ({skipped} infeasible size variants skipped) -> {args.out}")


if __name__ == "__main__":
    main()
