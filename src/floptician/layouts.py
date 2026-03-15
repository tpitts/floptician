from __future__ import annotations

import logging

from floptician.models import BoardConfiguration, CardDetection, CommunityCard, LayoutDescriptor, RowSpec

logger = logging.getLogger(__name__)

# ── Registry ──────────────────────────────────────────────────────────────────

_LAYOUT_REGISTRY: list[LayoutMatcher] = []


class LayoutMatcher:
    """Base class for layout matchers. Each subclass bundles matching, coordinate
    assignment, and validation for a single board configuration."""

    descriptor: LayoutDescriptor

    def configuration(self) -> BoardConfiguration:
        raise NotImplementedError

    def matches(self, rows: list[list[CardDetection]], all_detections: list[CardDetection]) -> tuple[bool, dict]:
        raise NotImplementedError

    def assign_coordinates(self, rows: list[list[CardDetection]], context: dict) -> list[CommunityCard]:
        raise NotImplementedError

    def validate(self, community_cards: list[CommunityCard]) -> bool:
        raise NotImplementedError


def register_layout(layout: LayoutMatcher) -> None:
    _LAYOUT_REGISTRY.append(layout)
    _LAYOUT_REGISTRY.sort(key=lambda m: m.descriptor.priority, reverse=True)


def get_layout(config: BoardConfiguration) -> LayoutMatcher | None:
    for layout in _LAYOUT_REGISTRY:
        if layout.configuration() == config:
            return layout
    return None


def all_layouts() -> list[LayoutMatcher]:
    return list(_LAYOUT_REGISTRY)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sorted_by_x(row: list[CardDetection]) -> list[CardDetection]:
    return sorted(row, key=lambda c: c.box.center_x)


def _make_cards(row: list[CardDetection], y: int, x_start: int = 1) -> list[CommunityCard]:
    return [
        CommunityCard(card=c.card, x=x_start + i, y=y, confidence=c.confidence)
        for i, c in enumerate(_sorted_by_x(row))
    ]


# ── SingleRow ─────────────────────────────────────────────────────────────────


class SingleRowLayout(LayoutMatcher):
    descriptor = LayoutDescriptor(
        name="SINGLE_ROW",
        row_specs=(RowSpec(role="main", valid_card_counts=(3, 4, 5), y_coordinate=1),),
        total_cards_range=(3, 5),
        min_detection_cards=3,
        priority=10,
    )

    def configuration(self) -> BoardConfiguration:
        return BoardConfiguration.SINGLE_ROW

    def matches(self, rows: list[list[CardDetection]], all_detections: list[CardDetection]) -> tuple[bool, dict]:
        # Match any set of rows where at least one has 3-5 cards.
        # Pick the largest qualifying row.
        best_row = None
        best_idx = -1
        for i, row in enumerate(rows):
            if 3 <= len(row) <= 5 and (best_row is None or len(row) > len(best_row)):
                best_row = row
                best_idx = i
        if best_row is not None:
            return True, {"main_row_idx": best_idx}
        return False, {}

    def assign_coordinates(self, rows: list[list[CardDetection]], context: dict) -> list[CommunityCard]:
        idx = context["main_row_idx"]
        return _make_cards(rows[idx], y=1)

    def validate(self, community_cards: list[CommunityCard]) -> bool:
        y_coords = {c.y for c in community_cards}
        return y_coords == {1} and len(community_cards) in (3, 4, 5)


# ── TwoRows ───────────────────────────────────────────────────────────────────


class TwoRowsLayout(LayoutMatcher):
    descriptor = LayoutDescriptor(
        name="TWO_ROWS",
        row_specs=(
            RowSpec(role="top", valid_card_counts=(3, 4, 5), y_coordinate=1),
            RowSpec(role="bottom", valid_card_counts=(3, 4, 5), y_coordinate=3),
        ),
        total_cards_range=(6, 10),
        min_detection_cards=6,
        priority=20,
    )

    def configuration(self) -> BoardConfiguration:
        return BoardConfiguration.TWO_ROWS

    def matches(self, rows: list[list[CardDetection]], all_detections: list[CardDetection]) -> tuple[bool, dict]:
        # Need exactly 2 rows with 3-5 cards each, equal count
        big_rows = [(i, row) for i, row in enumerate(rows) if 3 <= len(row) <= 5]
        if len(big_rows) < 2:
            return False, {}
        # Take the 2 biggest rows (or first 2 if same size)
        big_rows.sort(key=lambda x: len(x[1]), reverse=True)
        top_idx, top_row = big_rows[0]
        bot_idx, bot_row = big_rows[1]
        if len(top_row) != len(bot_row):
            return False, {}
        # Ensure top is actually above bottom (by center_y)
        top_cy = sum(c.box.center_y for c in top_row) / len(top_row)
        bot_cy = sum(c.box.center_y for c in bot_row) / len(bot_row)
        if top_cy > bot_cy:
            top_idx, bot_idx = bot_idx, top_idx
        return True, {"top_idx": top_idx, "bot_idx": bot_idx}

    def assign_coordinates(self, rows: list[list[CardDetection]], context: dict) -> list[CommunityCard]:
        top = rows[context["top_idx"]]
        bot = rows[context["bot_idx"]]
        return _make_cards(top, y=1) + _make_cards(bot, y=3)

    def validate(self, community_cards: list[CommunityCard]) -> bool:
        top = [c for c in community_cards if c.y == 1]
        bot = [c for c in community_cards if c.y == 3]
        mid = [c for c in community_cards if c.y == 2]
        if mid:
            return False
        return len(top) == len(bot) and len(top) in (3, 4, 5)


# ── Chihuahua ─────────────────────────────────────────────────────────────────


class ChihuahuaLayout(LayoutMatcher):
    descriptor = LayoutDescriptor(
        name="CHIHUAHUA",
        row_specs=(
            RowSpec(role="top", valid_card_counts=(4, 5), y_coordinate=1),
            RowSpec(role="chihuahua", valid_card_counts=(1,), y_coordinate=2),
            RowSpec(role="bottom", valid_card_counts=(4, 5), y_coordinate=3),
        ),
        total_cards_range=(9, 11),
        min_detection_cards=9,
        priority=50,
    )

    def configuration(self) -> BoardConfiguration:
        return BoardConfiguration.CHIHUAHUA

    def matches(self, rows: list[list[CardDetection]], all_detections: list[CardDetection]) -> tuple[bool, dict]:
        # Need 2 rows with 4-5 cards each, equal count, plus a chihuahua card between and to the right
        big_rows = [(i, row) for i, row in enumerate(rows) if 4 <= len(row) <= 5]
        if len(big_rows) < 2:
            return False, {}
        # Sort by y to get top/bottom
        big_rows.sort(key=lambda x: sum(c.box.center_y for c in x[1]) / len(x[1]))
        top_idx, top_row = big_rows[0]
        bot_idx, bot_row = big_rows[1]
        if len(top_row) != len(bot_row):
            return False, {}

        chihuahua = self._find_chihuahua_card([top_row, bot_row], all_detections)
        if chihuahua is None:
            return False, {}
        return True, {"top_idx": top_idx, "bot_idx": bot_idx, "chihuahua": chihuahua}

    def assign_coordinates(self, rows: list[list[CardDetection]], context: dict) -> list[CommunityCard]:
        top = rows[context["top_idx"]]
        bot = rows[context["bot_idx"]]
        chihuahua: CardDetection = context["chihuahua"]

        cards: list[CommunityCard] = []
        for y, row in ((1, top), (3, bot)):
            for x, c in enumerate(_sorted_by_x(row), start=1):
                x_coord = 6 if x == 5 else x
                cards.append(CommunityCard(card=c.card, x=x_coord, y=y, confidence=c.confidence))

        cards.append(CommunityCard(card=chihuahua.card, x=5, y=2, confidence=chihuahua.confidence))
        return cards

    def validate(self, community_cards: list[CommunityCard]) -> bool:
        top = [c for c in community_cards if c.y == 1]
        mid = [c for c in community_cards if c.y == 2]
        bot = [c for c in community_cards if c.y == 3]
        return len(top) == len(bot) and len(top) in (4, 5) and len(mid) == 1

    # ── chihuahua-specific helpers ────────────────────────────────────────────

    @staticmethod
    def _find_chihuahua_card(
        two_rows: list[list[CardDetection]], all_detections: list[CardDetection]
    ) -> CardDetection | None:
        main_row_cards = {id(c) for row in two_rows for c in row}
        candidates = [c for c in all_detections if id(c) not in main_row_cards]

        for card in candidates:
            if ChihuahuaLayout._is_between_rows(card, two_rows) and ChihuahuaLayout._is_to_the_right(card, two_rows):
                logger.debug(f"Chihuahua card identified: {card.card}")
                return card
        return None

    @staticmethod
    def _is_between_rows(card: CardDetection, rows: list[list[CardDetection]]) -> bool:
        top_edge = min(c.box.y1 for row in rows for c in row)
        bottom_edge = max(c.box.y2 for row in rows for c in row)
        return top_edge < card.box.center_y < bottom_edge

    @staticmethod
    def _is_to_the_right(card: CardDetection, rows: list[list[CardDetection]]) -> bool:
        main_rows = [row[:4] for row in rows if len(row) >= 4]
        if not main_rows:
            return False
        fourth_cards_center_x = sum(sorted(c.box.center_x for c in row)[3] for row in main_rows) / len(main_rows)
        return card.box.center_x > fourth_cards_center_x


# ── RunItTwiceFlop ────────────────────────────────────────────────────────────


class RunItTwiceFlopLayout(LayoutMatcher):
    descriptor = LayoutDescriptor(
        name="RUN_IT_TWICE_FLOP",
        row_specs=(
            RowSpec(role="runout_top", valid_card_counts=(1, 2), y_coordinate=1),
            RowSpec(role="main", valid_card_counts=(3,), y_coordinate=2),
            RowSpec(role="runout_bottom", valid_card_counts=(1, 2), y_coordinate=3),
        ),
        total_cards_range=(5, 7),
        min_detection_cards=5,
        priority=30,
    )

    def configuration(self) -> BoardConfiguration:
        return BoardConfiguration.RUN_IT_TWICE_FLOP

    def matches(self, rows: list[list[CardDetection]], all_detections: list[CardDetection]) -> tuple[bool, dict]:
        # Need a 3-card main row plus 1-2 card rows above and below with equal counts
        main_rows = [(i, row) for i, row in enumerate(rows) if len(row) == 3]
        if not main_rows:
            return False, {}

        for main_idx, main_row in main_rows:
            main_cy = sum(c.box.center_y for c in main_row) / len(main_row)
            median_height = self._median_height(all_detections)

            top_rows = []
            bot_rows = []
            for i, row in enumerate(rows):
                if i == main_idx:
                    continue
                if not (1 <= len(row) <= 2):
                    continue
                row_cy = sum(c.box.center_y for c in row) / len(row)
                if abs(row_cy - main_cy) > median_height * 3:
                    continue
                if row_cy < main_cy:
                    top_rows.append((i, row))
                else:
                    bot_rows.append((i, row))

            if top_rows and bot_rows:
                top_idx, top_row = top_rows[0]
                bot_idx, bot_row = bot_rows[0]
                if len(top_row) == len(bot_row):
                    return True, {"top_idx": top_idx, "main_idx": main_idx, "bot_idx": bot_idx}

        return False, {}

    def assign_coordinates(self, rows: list[list[CardDetection]], context: dict) -> list[CommunityCard]:
        top = rows[context["top_idx"]]
        main = rows[context["main_idx"]]
        bot = rows[context["bot_idx"]]
        x_start = len(main) + 1
        return _make_cards(top, y=1, x_start=x_start) + _make_cards(main, y=2) + _make_cards(bot, y=3, x_start=x_start)

    def validate(self, community_cards: list[CommunityCard]) -> bool:
        top = [c for c in community_cards if c.y == 1]
        mid = [c for c in community_cards if c.y == 2]
        bot = [c for c in community_cards if c.y == 3]
        return len(mid) == 3 and len(top) == len(bot) and len(top) in (1, 2)

    @staticmethod
    def _median_height(detections: list[CardDetection]) -> float:
        import statistics

        return statistics.median(d.box.height for d in detections)


# ── RunItTwiceTurn ────────────────────────────────────────────────────────────


class RunItTwiceTurnLayout(LayoutMatcher):
    descriptor = LayoutDescriptor(
        name="RUN_IT_TWICE_TURN",
        row_specs=(
            RowSpec(role="runout_top", valid_card_counts=(1,), y_coordinate=1),
            RowSpec(role="main", valid_card_counts=(4,), y_coordinate=2),
            RowSpec(role="runout_bottom", valid_card_counts=(1,), y_coordinate=3),
        ),
        total_cards_range=(6, 6),
        min_detection_cards=6,
        priority=40,
    )

    def configuration(self) -> BoardConfiguration:
        return BoardConfiguration.RUN_IT_TWICE_TURN

    def matches(self, rows: list[list[CardDetection]], all_detections: list[CardDetection]) -> tuple[bool, dict]:
        # Try clean 4+1+1 first, then try splitting a 5-card row
        result = self._try_clean_match(rows, all_detections)
        if result[0]:
            return result
        return self._try_split_match(rows, all_detections)

    def _try_clean_match(
        self, rows: list[list[CardDetection]], all_detections: list[CardDetection]
    ) -> tuple[bool, dict]:
        """Standard match: 4-card main row + 1-card rows above/below."""
        main_rows = [(i, row) for i, row in enumerate(rows) if len(row) == 4]
        if not main_rows:
            return False, {}

        for main_idx, main_row in main_rows:
            main_cy = sum(c.box.center_y for c in main_row) / len(main_row)
            median_height = RunItTwiceFlopLayout._median_height(all_detections)

            top_rows = []
            bot_rows = []
            for i, row in enumerate(rows):
                if i == main_idx:
                    continue
                if len(row) != 1:
                    continue
                row_cy = row[0].box.center_y
                if abs(row_cy - main_cy) > median_height * 3:
                    continue
                if row_cy < main_cy:
                    top_rows.append((i, row))
                else:
                    bot_rows.append((i, row))

            if top_rows and bot_rows:
                return True, {"top_idx": top_rows[0][0], "main_idx": main_idx, "bot_idx": bot_rows[0][0]}

        return False, {}

    def _try_split_match(
        self, rows: list[list[CardDetection]], all_detections: list[CardDetection]
    ) -> tuple[bool, dict]:
        """Handle case where a runout card got absorbed into a 5-card row.
        Look for a 5-card row + 1-card row, and try splitting the outlier card
        from the 5-card row to form 4+1+1."""
        five_card_rows = [(i, row) for i, row in enumerate(rows) if len(row) == 5]
        one_card_rows = [(i, row) for i, row in enumerate(rows) if len(row) == 1]

        if not five_card_rows or not one_card_rows:
            return False, {}

        median_height = RunItTwiceFlopLayout._median_height(all_detections)

        for five_idx, five_row in five_card_rows:
            # Find the card in the 5-card row most vertically offset from the other 4
            outlier_card, _outlier_pos = self._find_vertical_outlier(five_row, median_height)
            if outlier_card is None:
                continue

            remaining = [c for c in five_row if c is not outlier_card]
            main_cy = sum(c.box.center_y for c in remaining) / len(remaining)

            for one_idx, one_row in one_card_rows:
                single_cy = one_row[0].box.center_y
                if abs(single_cy - main_cy) > median_height * 3:
                    continue

                # One must be above, one below
                outlier_above = outlier_card.box.center_y < main_cy
                single_above = single_cy < main_cy
                if outlier_above == single_above:
                    continue  # Both on same side

                # Build virtual rows: we store the split info in context
                return True, {
                    "split_from": five_idx,
                    "outlier": outlier_card,
                    "one_card_idx": one_idx,
                    "main_cy": main_cy,
                }

        return False, {}

    @staticmethod
    def _find_vertical_outlier(
        row: list[CardDetection], median_height: float
    ) -> tuple[CardDetection | None, int]:
        """Find a card at the edge (first or last by x) that is vertically
        offset from the other cards. Returns (card, index) or (None, -1)."""
        sorted_row = sorted(row, key=lambda c: c.box.center_x)
        # Only check first and last cards (edges)
        for candidate_idx in (0, len(sorted_row) - 1):
            candidate = sorted_row[candidate_idx]
            others = [c for j, c in enumerate(sorted_row) if j != candidate_idx]
            others_cy = sum(c.box.center_y for c in others) / len(others)
            offset = abs(candidate.box.center_y - others_cy)
            # Must be vertically offset by at least 15% of card height
            if offset > median_height * 0.15:
                return candidate, candidate_idx
        return None, -1

    def assign_coordinates(self, rows: list[list[CardDetection]], context: dict) -> list[CommunityCard]:
        if "split_from" in context:
            return self._assign_split_coordinates(rows, context)
        top = rows[context["top_idx"]]
        main = rows[context["main_idx"]]
        bot = rows[context["bot_idx"]]
        x_start = len(main) + 1
        return _make_cards(top, y=1, x_start=x_start) + _make_cards(main, y=2) + _make_cards(bot, y=3, x_start=x_start)

    def _assign_split_coordinates(self, rows: list[list[CardDetection]], context: dict) -> list[CommunityCard]:
        five_row = rows[context["split_from"]]
        outlier: CardDetection = context["outlier"]
        one_row = rows[context["one_card_idx"]]
        main_cy = context["main_cy"]

        main = [c for c in five_row if c is not outlier]
        # Determine which is top/bottom
        outlier_above = outlier.box.center_y < main_cy
        single_above = one_row[0].box.center_y < main_cy

        if outlier_above:
            top_cards, bot_cards = [outlier], one_row
        else:
            top_cards, bot_cards = one_row, [outlier]

        # If single is above, swap
        if single_above and not outlier_above:
            top_cards, bot_cards = one_row, [outlier]
        elif not single_above and outlier_above:
            top_cards, bot_cards = [outlier], one_row

        x_start = len(main) + 1
        top = _make_cards(top_cards, y=1, x_start=x_start)
        mid = _make_cards(main, y=2)
        bot = _make_cards(bot_cards, y=3, x_start=x_start)
        return top + mid + bot

    def validate(self, community_cards: list[CommunityCard]) -> bool:
        top = [c for c in community_cards if c.y == 1]
        mid = [c for c in community_cards if c.y == 2]
        bot = [c for c in community_cards if c.y == 3]
        return len(mid) == 4 and len(top) == 1 and len(bot) == 1


# ── Auto-register all layouts ────────────────────────────────────────────────

register_layout(SingleRowLayout())
register_layout(TwoRowsLayout())
register_layout(ChihuahuaLayout())
register_layout(RunItTwiceFlopLayout())
register_layout(RunItTwiceTurnLayout())
