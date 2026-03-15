from __future__ import annotations

import logging
import statistics

from floptician.layouts import all_layouts
from floptician.models import BoardConfiguration, BoardProcessorConfig, CardDetection, CommunityCard

logger = logging.getLogger(__name__)


class CommunityCardDetector:
    def __init__(self, config: BoardProcessorConfig):
        self.config = config
        self.vertical_alignment_threshold = config.vertical_alignment_threshold
        self.horizontal_alignment_threshold = config.horizontal_alignment_threshold
        self.image_height = config.image_height

    def detect_community_cards(
        self, detections: list[CardDetection]
    ) -> tuple[list[CommunityCard], BoardConfiguration]:
        if len(detections) < 3:
            return [], BoardConfiguration.NO_BOARD

        rows = self._group_cards_into_rows(detections)
        rows = self._sort_rows_by_y(rows)

        for layout in all_layouts():
            matched, context = layout.matches(rows, detections)
            if matched:
                cards = layout.assign_coordinates(rows, context)
                logger.debug(f"Detected board configuration: {layout.configuration().name}")
                return cards, layout.configuration()

        return [], BoardConfiguration.NO_BOARD

    def _group_cards_into_rows(self, detections: list[CardDetection]) -> list[list[CardDetection]]:
        if not detections:
            return []

        widths = [d.box.width for d in detections]
        heights = [d.box.height for d in detections]
        median_width = statistics.median(widths)
        median_height = statistics.median(heights)

        # Step 1: Vertical grouping — group all detections (min_cards=1)
        centers = [(d, d.box.center_x, d.box.center_y) for d in detections]
        centers.sort(key=lambda c: c[2])

        vertical_groups: list[list[tuple[CardDetection, float, float]]] = []
        current_group = [centers[0]]
        for detection, cx, cy in centers[1:]:
            prev_cy = current_group[-1][2]
            vertical_threshold = median_height * self.vertical_alignment_threshold
            if abs(cy - prev_cy) < vertical_threshold:
                current_group.append((detection, cx, cy))
            else:
                vertical_groups.append(current_group)
                current_group = [(detection, cx, cy)]

        if current_group:
            vertical_groups.append(current_group)

        # Step 2: Horizontal refinement — split wide gaps, keep rows of 1+ cards
        all_rows: list[list[CardDetection]] = []
        for group in vertical_groups:
            group.sort(key=lambda c: c[1])
            current_row = [group[0]]
            for detection, cx, cy in group[1:]:
                _prev_det, prev_cx, _prev_cy = current_row[-1]
                horizontal_distance = cx - prev_cx

                if len(current_row) == 1:
                    horizontal_threshold = 1.75 * median_width
                elif len(current_row) >= 4:
                    horizontal_threshold = 3.25 * median_width
                else:
                    horizontal_threshold = 2.5 * median_width

                if horizontal_distance < horizontal_threshold:
                    current_row.append((detection, cx, cy))
                else:
                    all_rows.append([d for d, _, _ in current_row])
                    current_row = [(detection, cx, cy)]

            all_rows.append([d for d, _, _ in current_row])

        # Step 3: Two-phase filtering — keep small rows only if a main row exists nearby
        big_rows = [row for row in all_rows if len(row) >= 3]
        if not big_rows:
            logger.debug("No rows with 3+ cards found")
            return []

        # Compute center_y of the biggest row for proximity check
        main_row = max(big_rows, key=len)
        main_cy = sum(d.box.center_y for d in main_row) / len(main_row)
        proximity_limit = median_height * 3

        filtered_rows: list[list[CardDetection]] = []
        for row in all_rows:
            if len(row) >= 3:
                filtered_rows.append(row)
            else:
                row_cy = sum(d.box.center_y for d in row) / len(row)
                if abs(row_cy - main_cy) <= proximity_limit:
                    filtered_rows.append(row)

        logger.debug(f"Median card width: {median_width:.2f}, height: {median_height:.2f}")
        logger.debug(f"Number of detections: {len(detections)}")
        logger.debug(f"Number of vertical groups: {len(vertical_groups)}")
        logger.debug(f"Number of final rows: {len(filtered_rows)}")

        for i, row in enumerate(filtered_rows):
            logger.debug(f"Row {i + 1}: {len(row)} cards")
            logger.debug(f"  Card centers: {[(d.box.center_x, d.box.center_y) for d in row]}")

        return filtered_rows

    def _sort_rows_by_y(self, rows: list[list[CardDetection]]) -> list[list[CardDetection]]:
        """Sort rows top-to-bottom by average y coordinate."""
        if len(rows) <= 1:
            return rows
        return sorted(rows, key=lambda row: sum(d.box.center_y for d in row) / len(row))

    def validate_configuration(self, community_cards: list[CommunityCard]) -> bool:
        return any(layout.validate(community_cards) for layout in all_layouts())
