from __future__ import annotations

import logging
import statistics

from floptician.models import BoardConfiguration, BoardProcessorConfig, CardDetection, CommunityCard

logger = logging.getLogger(__name__)


class CommunityCardDetector:
    def __init__(self, config: BoardProcessorConfig):
        self.config = config
        self.min_cards_in_row = 3
        self.vertical_alignment_threshold = config.vertical_alignment_threshold
        self.horizontal_alignment_threshold = config.horizontal_alignment_threshold
        self.image_height = config.image_height

    def detect_community_cards(self, detections: list[CardDetection]) -> list[CommunityCard]:
        if len(detections) < 3:
            return []

        rows = self._group_cards_into_rows(detections)
        rows = self._filter_and_sort_rows(rows)
        configuration, chihuahua_card = self._determine_configuration(rows, detections)

        logger.debug(f"Detected board configuration: {configuration.name}")

        return self._assign_coordinates(rows, configuration, chihuahua_card)

    def _group_cards_into_rows(self, detections: list[CardDetection]) -> list[list[CardDetection]]:
        if not detections:
            return []

        # Calculate median width and height for adaptive thresholding
        widths = [d.box.width for d in detections]
        heights = [d.box.height for d in detections]
        median_width = statistics.median(widths)
        median_height = statistics.median(heights)

        # Step 1: Vertical Grouping
        centers = [(d, d.box.center_x, d.box.center_y) for d in detections]
        centers.sort(key=lambda c: c[2])  # Sort by y-coordinate

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

        # Step 2: Horizontal Refinement within Vertical Groups
        rows: list[list[CardDetection]] = []
        for group in vertical_groups:
            group.sort(key=lambda c: c[1])  # Sort by x-coordinate
            current_row = [group[0]]
            for detection, cx, cy in group[1:]:
                _prev_det, prev_cx, _prev_cy = current_row[-1]
                horizontal_distance = cx - prev_cx

                # Adaptive horizontal thresholding
                if len(current_row) == 1:
                    horizontal_threshold = 1.75 * median_width
                elif len(current_row) >= 4:
                    horizontal_threshold = 3.25 * median_width
                else:
                    horizontal_threshold = 2.5 * median_width

                if horizontal_distance < horizontal_threshold:
                    current_row.append((detection, cx, cy))
                else:
                    if len(current_row) >= self.min_cards_in_row:
                        rows.append([d for d, _, _ in current_row])
                    current_row = [(detection, cx, cy)]

            if len(current_row) >= self.min_cards_in_row:
                rows.append([d for d, _, _ in current_row])

        # Logging and debugging
        logger.debug(f"Median card width: {median_width:.2f}, height: {median_height:.2f}")
        logger.debug(f"Number of detections: {len(detections)}")
        logger.debug(f"Number of vertical groups: {len(vertical_groups)}")
        logger.debug(f"Number of final rows: {len(rows)}")

        for i, row in enumerate(rows):
            logger.debug(f"Row {i + 1}: {len(row)} cards")
            logger.debug(f"  Card centers: {[(d.box.center_x, d.box.center_y) for d in row]}")

        return rows

    def _filter_and_sort_rows(self, rows: list[list[CardDetection]]) -> list[list[CardDetection]]:
        if len(rows) <= 2:
            return rows

        image_center = self.image_height / 2
        row_centers = [(row, sum(d.box.y1 + d.box.y2 for d in row) / (2 * len(row))) for row in rows]
        sorted_rows = sorted(row_centers, key=lambda x: abs(x[1] - image_center))
        return [row for row, _ in sorted_rows[:2]]

    def _determine_configuration(
        self, rows: list[list[CardDetection]], all_detections: list[CardDetection]
    ) -> tuple[BoardConfiguration, CardDetection | None]:
        if not rows:
            return BoardConfiguration.NO_BOARD, None

        if len(rows) == 1:
            return BoardConfiguration.SINGLE_ROW, None
        else:
            chihuahua_card = self._find_chihuahua_card(rows, all_detections)
            if chihuahua_card:
                return BoardConfiguration.CHIHUAHUA, chihuahua_card
            elif len(rows) >= 2:
                return BoardConfiguration.TWO_ROWS, None

        return BoardConfiguration.NO_BOARD, None

    def _find_chihuahua_card(
        self, rows: list[list[CardDetection]], all_detections: list[CardDetection]
    ) -> CardDetection | None:
        if len(rows) < 2:
            return None

        main_row_cards = [card for row in rows[:2] for card in row]
        potential_chihuahua_cards = [card for card in all_detections if card not in main_row_cards]

        for card in potential_chihuahua_cards:
            if self._is_between_rows(card, rows[:2]):
                logger.debug(f"Card {card.card} passed vertical positioning check")
                if self._is_to_the_right(card, rows[:2]):
                    logger.debug(f"Card {card.card} passed horizontal positioning check")
                    if all(len(row) in (4, 5) for row in rows[:2]) and len(rows[0]) == len(rows[1]):
                        logger.debug(f"Card {card.card} passed main rows length check. Identified as Chihuahua card.")
                        return card

        logger.debug("No card identified as a Chihuahua card")
        return None

    def _is_between_rows(self, card: CardDetection, rows: list[list[CardDetection]]) -> bool:
        card_center_y = card.box.center_y
        top_edge = min(c.box.y1 for row in rows for c in row)
        bottom_edge = max(c.box.y2 for row in rows for c in row)

        return top_edge < card_center_y < bottom_edge

    def _is_to_the_right(self, card: CardDetection, rows: list[list[CardDetection]]) -> bool:
        card_center_x = card.box.center_x
        main_rows = [row[:4] for row in rows[:2] if len(row) >= 4]

        if not main_rows:
            logger.debug(f"Card {card.card} failed 'to the right' check: No valid main rows found")
            return False

        fourth_cards_center_x = sum(row[3].box.center_x for row in main_rows) / len(main_rows)
        is_to_right = card_center_x > fourth_cards_center_x

        logger.debug(
            f"Card {card.card} center X: {card_center_x:.2f}, "
            f"4th cards average center X: {fourth_cards_center_x:.2f}, "
            f"Is to the right: {is_to_right}"
        )

        return is_to_right

    def _assign_coordinates(
        self,
        rows: list[list[CardDetection]],
        configuration: BoardConfiguration,
        chihuahua_card: CardDetection | None,
    ) -> list[CommunityCard]:
        community_cards: list[CommunityCard] = []

        if configuration == BoardConfiguration.NO_BOARD:
            return community_cards

        for y, row in enumerate(rows[:2], start=1):
            sorted_row = sorted(row, key=lambda card: card.box.center_x)

            for x, card in enumerate(sorted_row, start=1):
                x_coord = x
                if configuration == BoardConfiguration.CHIHUAHUA and x == 5:
                    x_coord = 6
                y_coord = y if y == 1 or configuration == BoardConfiguration.SINGLE_ROW else 3
                community_cards.append(CommunityCard(card=card.card, x=x_coord, y=y_coord, confidence=card.confidence))

        if configuration == BoardConfiguration.CHIHUAHUA and chihuahua_card:
            community_cards.append(
                CommunityCard(card=chihuahua_card.card, x=5, y=2, confidence=chihuahua_card.confidence)
            )

        return community_cards

    def validate_configuration(self, community_cards: list[CommunityCard]) -> bool:
        x_coords = set(card.x for card in community_cards)
        y_coords = set(card.y for card in community_cards)

        if len(y_coords) == 1:
            return len(x_coords) in (3, 4, 5)
        elif len(y_coords) == 2:
            top_row = [card for card in community_cards if card.y == 1]
            bottom_row = [card for card in community_cards if card.y == 3]
            middle_card = [card for card in community_cards if card.y == 2]

            if middle_card:
                return len(top_row) == len(bottom_row) and len(top_row) in (4, 5) and len(middle_card) == 1
            else:
                return len(top_row) == len(bottom_row) and len(top_row) in (3, 4, 5)
        else:
            return False
