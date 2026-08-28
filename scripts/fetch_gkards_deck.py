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

# Standard poker card corner radius is ~3.5mm on a 63mm-wide card
CORNER_RADIUS_FRAC = 0.055


def download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise
    # gkards serves an HTML page with HTTP 200 for images that don't exist
    if not data.startswith(b"\xff\xd8"):
        return False
    dest.write_bytes(data)
    return True


def rounded_corner_alpha(width: int, height: int) -> np.ndarray:
    """Antialiased rounded-rectangle alpha mask (drawn at 4x and downsampled)."""
    scale = 4
    radius = int(round(min(width, height) * CORNER_RADIUS_FRAC)) * scale
    mask = Image.new("L", (width * scale, height * scale), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, width * scale - 1, height * scale - 1], radius=radius, fill=255)
    mask = mask.resize((width, height), Image.LANCZOS)
    arr = np.array(mask)
    arr[arr < 8] = 0  # kill downsampling ringing so corners are fully transparent
    return arr


def cutout_card(src: Path, dest: Path) -> str:
    """Convert a gkards scan JPEG to a transparent PNG.

    gkards scans are already tightly cropped to the card, so the whole frame is
    the card face; only the corners (and any sliver of scanner background) need
    to be made transparent, which the rounded-rectangle alpha mask handles.
    Contour-based detection was tried and rejected: white card borders blend
    into the scan background, so it locks onto the inner artwork and over-crops.

    Returns a status string: 'ok' or 'bad-aspect' (result doesn't look card-shaped).
    """
    bgr = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not read {src}")

    if bgr.shape[1] > bgr.shape[0]:  # cards are portrait
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)

    h, w = bgr.shape[:2]
    status = "ok" if 0.55 <= w / h <= 0.85 else "bad-aspect"

    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = rounded_corner_alpha(w, h)
    Image.fromarray(rgba).save(dest)
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
