"""Download a deck's card scans from gkards.com and convert them to transparent PNGs.

Usage:
    python scripts/fetch_gkards_deck.py 000454 kem-deck-1
    python scripts/fetch_gkards_deck.py 000338 congress-606 --backs-dir resources/gkards-raw/backs

Card images land in resources/decks/<deck_name>/ using the repo naming convention
(<rank><suit>.png, e.g. As.png, Th.png). Raw JPEGs are cached in
resources/gkards-raw/deck-<id>/ so re-runs skip the download. The deck's back is
processed into the raw dir (and --backs-dir if given) but NOT into the deck folder,
since backs live at the decks root as back<N>.png.
"""

import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from _common import resources_path

BASE_URL = "https://www.gkards.com/images/deck-{id}/deck-{id}-{code}.jpg"
USER_AGENT = "floptician-deck-fetcher/1.0 (personal ML training data; contact: local use)"
REQUEST_DELAY_S = 0.5

SUITS = {"spad": "s", "hear": "h", "club": "c", "diam": "d"}
RANKS = {"A": "A", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
         "7": "7", "8": "8", "9": "9", "10": "T", "J": "J", "Q": "Q", "K": "K"}

# Fallback corner radius when it can't be measured (fraction of card width)
CORNER_RADIUS_FRAC_DEFAULT = 0.065
# How far in from each frame border the card edge may plausibly sit
MAX_MARGIN_FRAC = 0.05
# Mean Sobel response a column/row must reach to count as the card's edge line
EDGE_GRADIENT_MIN = 8.0
# Color distance from scanner background that counts as "card"
BG_DIST_THRESH = 20.0
# Shave this many px off every side so edge shadow/fringe lands outside the mask
EDGE_INSET_PX = 2


def download(url: str, dest: Path, retries: int = 3) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            if attempt == retries - 1:
                print(f"  giving up on {url} (HTTP {e.code}); re-run later to resume")
                return False
            time.sleep(20 * (attempt + 1))
        except (TimeoutError, urllib.error.URLError, OSError):
            # server throttling; back off, then skip (the run is resumable)
            if attempt == retries - 1:
                print(f"  giving up on {url} (timeout); re-run later to resume")
                return False
            time.sleep(20 * (attempt + 1))
    else:
        return False
    # gkards serves an HTML page with HTTP 200 for images that don't exist
    if not data.startswith(b"\xff\xd8"):
        return False
    dest.write_bytes(data)
    return True


def _edge_offset(grad_means: np.ndarray) -> int:
    """Index just inside the strongest gradient line near a border, or 0 if none."""
    if grad_means.size == 0:
        return 0
    peak = int(np.argmax(grad_means))
    if grad_means[peak] < EDGE_GRADIENT_MIN:
        return 0
    return peak + 2


def tight_crop(bgr: np.ndarray) -> np.ndarray:
    """Crop away the scanner-background margin around the card.

    The card edge shows up as a line of strong gradient (shadow/outline) close to
    each frame border, even when card and background are both near-white. Only the
    middle band of each side is sampled so the corner curvature doesn't dilute it.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    mx = max(3, int(w * MAX_MARGIN_FRAC))
    my = max(3, int(h * MAX_MARGIN_FRAC))
    rows = slice(int(h * 0.2), int(h * 0.8))
    cols = slice(int(w * 0.2), int(w * 0.8))
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    left = _edge_offset(gx[rows, :mx].mean(axis=0))
    right = _edge_offset(gx[rows, w - mx:].mean(axis=0)[::-1])
    top = _edge_offset(gy[:my, cols].mean(axis=1))
    bottom = _edge_offset(gy[h - my:, cols].mean(axis=1)[::-1])
    return bgr[top:h - bottom, left:w - right]


def estimate_corner_radius(bgr: np.ndarray) -> float:
    """Measure the card's corner radius from the background wedges at its corners.

    Walks each corner's diagonal until the pixel stops matching that corner's
    background color; for a quarter-circle of radius r the diagonal crosses the
    arc at r * (1 - 1/sqrt(2)) from the corner. Median of the usable corners.
    """
    h, w = bgr.shape[:2]
    p = max(3, min(h, w) // 60)
    reach = int(min(h, w) * 0.25)
    corners = [(0, 0, 1, 1), (0, w - 1, 1, -1), (h - 1, 0, -1, 1), (h - 1, w - 1, -1, -1)]
    radii = []
    for y0, x0, dy, dx in corners:
        patch = bgr[y0:y0 + dy * p:dy, x0:x0 + dx * p:dx].reshape(-1, 3)
        bg = np.median(patch, axis=0)
        for i in range(reach):
            px = bgr[y0 + dy * i, x0 + dx * i].astype(np.float32)
            if np.linalg.norm(px - bg) > BG_DIST_THRESH:
                if i >= 3:  # 0-2 px means dust or no measurable wedge
                    radii.append(i / (1 - 1 / np.sqrt(2)))
                break
    if not radii:
        return CORNER_RADIUS_FRAC_DEFAULT * min(w, h)
    r = float(np.median(radii))
    return float(np.clip(r, 0.04 * min(w, h), 0.18 * min(w, h)))


def rounded_corner_alpha(width: int, height: int, radius: float, inset: int = EDGE_INSET_PX) -> np.ndarray:
    """Antialiased rounded-rectangle alpha mask (drawn at 4x and downsampled)."""
    scale = 4
    mask = Image.new("L", (width * scale, height * scale), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        [inset * scale, inset * scale, (width - inset) * scale - 1, (height - inset) * scale - 1],
        radius=int(round(radius * scale)), fill=255)
    mask = mask.resize((width, height), Image.LANCZOS)
    arr = np.array(mask)
    arr[arr < 8] = 0  # kill downsampling ringing so corners are fully transparent
    return arr


def card_to_rgba(bgr: np.ndarray) -> np.ndarray:
    """Shared finishing step: measure the corner radius and apply the alpha mask."""
    h, w = bgr.shape[:2]
    radius = estimate_corner_radius(bgr)
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = rounded_corner_alpha(w, h, radius)
    return rgba


def cutout_card(src: Path, dest: Path) -> str:
    """Convert a gkards scan JPEG to a tightly-cropped transparent PNG.

    gkards scans are cropped close to the card but keep a few pixels of scanner
    background on each side; tight_crop removes that, then the alpha mask (with
    per-card measured corner radius) makes the corners transparent. Full contour
    detection was tried and rejected: white card borders blend into the scan
    background, so it locks onto the inner artwork and over-crops.

    Returns a status string: 'ok' or 'bad-aspect' (result doesn't look card-shaped).
    """
    bgr = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not read {src}")

    bgr = tight_crop(bgr)
    if bgr.shape[1] > bgr.shape[0]:  # cards are portrait
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)

    h, w = bgr.shape[:2]
    status = "ok" if 0.55 <= w / h <= 0.85 else "bad-aspect"

    Image.fromarray(card_to_rgba(bgr)).save(dest)
    return status


def fetch_deck(deck_id: str, deck_name: str, backs_dir: Path | None) -> None:
    raw_dir = resources_path("gkards-raw", f"deck-{deck_id}")
    out_dir = resources_path("decks", deck_name)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = {f"{gs}{gr}": f"{rr}{rs}" for gs, rs in SUITS.items() for gr, rr in RANKS.items()}
    codes["back"] = None

    missing, flagged = [], []
    for code, card in codes.items():
        raw = raw_dir / f"{code}.jpg"
        if not raw.exists():
            url = BASE_URL.format(id=deck_id, code=code)
            if not download(url, raw):
                missing.append(code)
                continue
            time.sleep(REQUEST_DELAY_S)
        if card is None:
            dest = raw_dir / "back.png"
            status = cutout_card(raw, dest)
            if backs_dir is not None:
                backs_dir.mkdir(parents=True, exist_ok=True)
                cutout_card(raw, backs_dir / f"{deck_name}-back.png")
        else:
            dest = out_dir / f"{card}.png"
            status = cutout_card(raw, dest)
        if status != "ok":
            flagged.append(dest.name)

    got = len([c for c in codes if c != "back" and c not in missing])
    print(f"[{deck_name}] {got}/52 cards -> {out_dir}")
    if missing:
        print(f"[{deck_name}] not on gkards (404): {', '.join(sorted(missing))}")
    if flagged:
        print(f"[{deck_name}] suspicious aspect ratio (check manually): {', '.join(sorted(flagged))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a gkards.com deck into resources/decks/.")
    parser.add_argument("deck_id", help="gkards deck id, e.g. 000454")
    parser.add_argument("deck_name", help="output folder name under resources/decks/, e.g. kem-deck-1")
    parser.add_argument("--backs-dir", type=Path, default=None,
                        help="also copy the processed card back into this directory")
    args = parser.parse_args()
    fetch_deck(args.deck_id, args.deck_name, args.backs_dir)


if __name__ == "__main__":
    main()
