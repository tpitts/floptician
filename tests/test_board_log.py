import io
import logging

import pytest
from rich.console import Console

from floptician.board_log import format_board
from floptician.logging_config import BoardLogHandler
from floptician.models import BoardConfiguration, CommunityCard


def test_four_suits_and_ten_labels():
    board = format_board(
        BoardConfiguration.SINGLE_ROW,
        [CommunityCard(card, i, 1) for i, card in enumerate(["Ah", "Td", "Kc", "Qs"], 1)],
    )
    assert board.plain == "Board confirmed · Single Row\n  A♥   10♦  K♣   Q♠"
    assert [(board.plain[s.start : s.end], s.style) for s in board.spans] == [
        ("A♥", "bold bright_red"),
        ("10♦", "bold #38bdf8"),
        ("K♣", "bold bright_green"),
        ("Q♠", "bold #6366f1"),
    ]


@pytest.mark.parametrize("layout", [BoardConfiguration.CHIHUAHUA, BoardConfiguration.RUN_IT_TWICE_TURN])
def test_rows_and_column_gaps_preserve_positions(layout):
    board = format_board(layout, [CommunityCard("Qs", 5, 3), CommunityCard("Ah", 1, 2), CommunityCard("Td", 5, 1)])
    lines = board.plain.splitlines()
    assert lines[1] == " " * 22 + "10♦"
    assert lines[2] == "  A♥"
    assert lines[3] == " " * 22 + "Q♠"


@pytest.mark.parametrize("terminal", [True, False])
def test_handler_colors_terminal_but_not_redirected_output(terminal, monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    output = io.StringIO()
    console = Console(file=output, force_terminal=terminal, width=100)
    handler = BoardLogHandler(console=console, show_time=False, show_level=False, show_path=False)
    board = format_board(BoardConfiguration.SINGLE_ROW, [CommunityCard("Qs", 1, 1)])
    record = logging.LogRecord("test", logging.INFO, "", 0, board.plain, (), None)
    record.board_text = board
    handler.emit(record)
    result = output.getvalue()
    assert "Q♠" in result
    assert ("\x1b[" in result) == terminal
    assert "bright_magenta" not in result
    assert record.getMessage() == board.plain
