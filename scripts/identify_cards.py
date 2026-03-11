"""Identify and rename card images using OpenAI Vision API."""

import argparse
import base64
import json
import sys
from collections import defaultdict
from pathlib import Path

from _common import SCRIPT_DIR  # noqa: F401 — ensures repo root on sys.path

SECRETS_PATH = Path(
    r"C:\Users\tompi\projects\openwakeword-recognizer\secrets\openwakeword_tts.json"
)

VALID_RANKS = {"A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"}
VALID_SUITS = {"s", "h", "d", "c"}

PROMPT = """\
Look at this image of a playing card extracted from a poker stream.
Identify the rank and suit of the card.

Respond with ONLY a JSON object, no markdown fences:
- If it is a recognizable playing card face: {"rank": "X", "suit": "y"}
  where rank is one of: A, 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K
  and suit is one of: s (spades), h (hearts), d (diamonds), c (clubs)
- If it is NOT a recognizable card (card back, chip, table felt, etc.): {"card": null}

Examples:
  Ace of spades -> {"rank": "A", "suit": "s"}
  Ten of hearts -> {"rank": "T", "suit": "h"}
  Not a card -> {"card": null}
"""


def load_api_key() -> str:
    with open(SECRETS_PATH) as f:
        return json.load(f)["openai_api_key"]


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def identify_card(client, image_path: Path) -> tuple[str, str] | None:
    """Return (rank, suit) or None if not a card."""
    b64 = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        max_completion_tokens=200,
    )
    raw = response.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  WARNING: Could not parse response: {raw}")
        return None

    if data.get("card") is None and "rank" not in data:
        return None

    rank = data.get("rank")
    suit = data.get("suit")
    if rank in VALID_RANKS and suit in VALID_SUITS:
        return (rank, suit)

    print(f"  WARNING: Invalid rank/suit: {data}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Identify and rename card images")
    parser.add_argument("input_dir", type=Path, help="Directory with card images")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print renames without executing"
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(api_key=load_api_key())

    images = sorted(input_dir.glob("*.png"))
    print(f"Found {len(images)} images in {input_dir}")

    counter: dict[str, int] = defaultdict(int)
    renames: list[tuple[Path, Path]] = []
    deletes: list[Path] = []

    for i, img in enumerate(images):
        safe_name = img.name[:80].encode("ascii", errors="replace").decode("ascii")
        print(f"[{i + 1}/{len(images)}] {safe_name}...", end=" ")
        try:
            result = identify_card(client, img)
        except Exception as e:
            print(f"  ERROR: {e}")
            deletes.append(img)
            continue
        if result is None:
            print("-> NOT A CARD (will delete)")
            deletes.append(img)
        else:
            rank, suit = result
            card_id = f"{rank}{suit}"
            idx = counter[card_id]
            counter[card_id] += 1
            new_name = f"{card_id}-{idx}.png"
            new_path = input_dir / new_name
            print(f"-> {new_name}")
            renames.append((img, new_path))

    print(f"\nSummary: {len(renames)} cards identified, {len(deletes)} non-cards")
    for card_id in sorted(counter):
        print(f"  {card_id}: {counter[card_id]} images")

    if args.dry_run:
        print("DRY RUN -- no changes made")
        return

    # Delete non-cards first
    for path in deletes:
        try:
            path.unlink()
        except OSError as e:
            print(f"  WARNING: Could not delete {path.name}: {e}")

    # Two-pass rename to avoid collisions (old name might match a new name)
    temps = []
    for old, new in renames:
        tmp = old.parent / (old.stem + ".tmp_rename")
        old.rename(tmp)
        temps.append((tmp, new))

    for tmp, new in temps:
        tmp.rename(new)

    print("Done!")


if __name__ == "__main__":
    main()
