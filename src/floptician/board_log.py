"""Readable board snapshots for terminal and plain-text logging."""

from collections.abc import Sequence

from rich.text import Text

from floptician.models import BoardConfiguration, CommunityCard

SUITS = {
    "h": ("♥", "bold bright_red"),
    "d": ("♦", "bold #38bdf8"),
    "c": ("♣", "bold bright_green"),
    "s": ("♠", "bold #6366f1"),
}


def format_board(configuration: BoardConfiguration, cards: Sequence[CommunityCard]) -> Text:
    """Preserve occupied rows and column gaps; style labels, never log markup."""
    text = Text(f"Board confirmed · {configuration.name.replace('_', ' ').title()}")
    for row in sorted({card.y for card in cards}):
        text.append("\n  ")
        column = 1
        for card in sorted((card for card in cards if card.y == row), key=lambda card: card.x):
            text.append(" " * max(0, card.x - column) * 5)
            rank, suit = card.card[:-1].upper(), card.card[-1:].lower()
            symbol, style = SUITS.get(suit, (suit, ""))
            label = f"{'10' if rank == 'T' else rank}{symbol}"
            text.append(label, style=style)
            text.append(" " * max(1, 5 - len(label)))
            column = card.x + 1
        text.rstrip()
    return text
