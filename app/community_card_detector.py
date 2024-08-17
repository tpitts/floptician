import logging
from typing import List, Dict, Tuple, Optional
from enum import Enum, auto

logger = logging.getLogger(__name__)

class BoardConfiguration(Enum):
    NO_BOARD = auto()
    SINGLE_ROW = auto()
    TWO_ROWS = auto()
    CHIHUAHUA = auto()

class CommunityCardDetector:
    def __init__(self, config: Dict):
        self.config = config
        self.min_cards_in_row = 3
        self.vertical_alignment_threshold = config.get('vertical_alignment_threshold', 0.2)
        self.horizontal_alignment_threshold = config.get('horizontal_alignment_threshold', 0.1)
        self.image_height = config.get('image_height', 1080)  # Assume 1080p if not specified

    def detect_community_cards(self, detections: List[Dict]) -> List[Dict]:
        """
        Main method to detect and process community cards.
        
        Args:
            detections (List[Dict]): List of detected card dictionaries, each containing 'box' and 'card' keys.
        
        Returns:
            List[Dict]: Processed community cards with assigned coordinates, or an empty list if no valid configuration is found.
        """
        if len(detections) < 3:
            return []

        rows = self._group_cards_into_rows(detections)
        rows = self._filter_and_sort_rows(rows)
        configuration, chihuahua_card = self._determine_configuration(rows, detections)
        
        logger.debug(f"Detected board configuration: {configuration.name}")
        
        return self._assign_coordinates(rows, configuration, chihuahua_card)

    def _group_cards_into_rows(self, detections: List[Dict]) -> List[List[Dict]]:
        # Group cards into rows based on vertical proximity
        def calculate_center(box):
            x1, y1, x2, y2 = box
            return [(x1 + x2) / 2, (y1 + y2) / 2]

        centers = [(d, calculate_center(d['box'])) for d in detections]
        centers.sort(key=lambda c: c[1][1])

        rows = []
        current_row = [centers[0]]

        for detection, center in centers[1:]:
            prev_center = current_row[-1][1]
            box_height = detection['box'][3] - detection['box'][1]
            threshold = box_height * self.vertical_alignment_threshold

            if abs(center[1] - prev_center[1]) < threshold:
                current_row.append((detection, center))
            else:
                if len(current_row) >= self.min_cards_in_row:
                    rows.append([d for d, _ in sorted(current_row, key=lambda x: x[1][0])])
                current_row = [(detection, center)]

        if len(current_row) >= self.min_cards_in_row:
            rows.append([d for d, _ in sorted(current_row, key=lambda x: x[1][0])])

        return rows

    def _filter_and_sort_rows(self, rows: List[List[Dict]]) -> List[List[Dict]]:
        # Keep only the two rows closest to the image center if more than two are detected
        if len(rows) <= 2:
            return rows

        image_center = self.image_height / 2
        row_centers = [(row, sum(d['box'][1] + d['box'][3] for d in row) / (2 * len(row))) for row in rows]
        sorted_rows = sorted(row_centers, key=lambda x: abs(x[1] - image_center))
        return [row for row, _ in sorted_rows[:2]]

    def _determine_configuration(self, rows: List[List[Dict]], all_detections: List[Dict]) -> Tuple[BoardConfiguration, Optional[Dict]]:
        # Determine the board configuration and identify a potential Chihuahua card
        if not rows:
            return BoardConfiguration.NO_BOARD, None

        if len(rows) == 1:
            return BoardConfiguration.SINGLE_ROW, None
        else:  # Two or more rows
            chihuahua_card = self._find_chihuahua_card(rows, all_detections)
            if chihuahua_card:
                return BoardConfiguration.CHIHUAHUA, chihuahua_card
            elif len(rows) >= 2:
                return BoardConfiguration.TWO_ROWS, None
        
        return BoardConfiguration.NO_BOARD, None

    def _find_chihuahua_card(self, rows: List[List[Dict]], all_detections: List[Dict]) -> Optional[Dict]:
        # Identify a potential Chihuahua card from all detections, not just those in rows
        if len(rows) < 2:
            return None

        main_row_cards = [card for row in rows[:2] for card in row]
        potential_chihuahua_cards = [card for card in all_detections if card not in main_row_cards]

        for card in potential_chihuahua_cards:
            if self._is_between_rows(card, rows[:2]):
                logger.debug(f"Card {card['card']} passed vertical positioning check")
                if self._is_to_the_right(card, rows[:2]):
                    logger.debug(f"Card {card['card']} passed horizontal positioning check")
                    if all(len(row) in (4, 5) for row in rows[:2]) and len(rows[0]) == len(rows[1]):
                        logger.debug(f"Card {card['card']} passed main rows length check. Identified as Chihuahua card.")
                        return card

        logger.debug("No card identified as a Chihuahua card")
        return None

    def _is_between_rows(self, card: Dict, rows: List[List[Dict]]) -> bool:
        # Check if a card's center is vertically between the top and bottom of all cards in the rows
        card_center_y = (card['box'][1] + card['box'][3]) / 2
        top_edge = min(c['box'][1] for row in rows for c in row)
        bottom_edge = max(c['box'][3] for row in rows for c in row)
        
        return top_edge < card_center_y < bottom_edge

    def _is_to_the_right(self, card: Dict, rows: List[List[Dict]]) -> bool:
        # Check if a card is to the right of the first 4 cards in each row
        card_center_x = (card['box'][0] + card['box'][2]) / 2

        # Consider only the first two rows and up to 4 cards in each row
        main_rows = [row[:4] for row in rows[:2] if len(row) >= 4]

        if not main_rows:
            logger.debug(f"Card {card['card']} failed 'to the right' check: No valid main rows found")
            return False

        # Calculate the average center of the 4th card in each row
        fourth_cards_center_x = sum((row[3]['box'][0] + row[3]['box'][2]) / 2 for row in main_rows) / len(main_rows)

        is_to_right = card_center_x > fourth_cards_center_x

        logger.debug(f"Card {card['card']} center X: {card_center_x:.2f}, "
                     f"4th cards average center X: {fourth_cards_center_x:.2f}, "
                     f"Is to the right: {is_to_right}")

        return is_to_right

    def _assign_coordinates(self, rows: List[List[Dict]], configuration: BoardConfiguration, chihuahua_card: Optional[Dict]) -> List[Dict]:
        # Assign x and y coordinates to cards based on their position and the board configuration
        community_cards = []

        if configuration == BoardConfiguration.NO_BOARD:
            return community_cards

        for y, row in enumerate(rows[:2], start=1):
            for x, card in enumerate(row, start=1):
                x_coord = x
                # Only in Chihuahua configuration, 5th card gets x=6
                if configuration == BoardConfiguration.CHIHUAHUA and x == 5:
                    x_coord = 6
                y_coord = y if y == 1 or configuration == BoardConfiguration.SINGLE_ROW else 3
                community_cards.append({
                    "card": card['card'],
                    "x": x_coord,
                    "y": y_coord,
                    "confidence": card.get('confidence', 1.0)
                })

        # Chihuahua card always gets x=5 and y=2
        if configuration == BoardConfiguration.CHIHUAHUA and chihuahua_card:
            community_cards.append({
                "card": chihuahua_card['card'],
                "x": 5,
                "y": 2,
                "confidence": chihuahua_card.get('confidence', 1.0)
            })

        return community_cards

    def validate_configuration(self, community_cards: List[Dict]) -> bool:
        # Validate the detected configuration based on the number and position of cards
        x_coords = set(card['x'] for card in community_cards)
        y_coords = set(card['y'] for card in community_cards)
        
        if len(y_coords) == 1:  # Single row
            return len(x_coords) in (3, 4, 5)
        elif len(y_coords) == 2:  # Two rows or Chihuahua
            top_row = [card for card in community_cards if card['y'] == 1]
            bottom_row = [card for card in community_cards if card['y'] == 3]
            middle_card = [card for card in community_cards if card['y'] == 2]
            
            if middle_card:  # Chihuahua
                return len(top_row) == len(bottom_row) and len(top_row) in (4, 5) and len(middle_card) == 1
            else:  # Two rows
                return len(top_row) == len(bottom_row) and len(top_row) in (3, 4, 5)
        else:
            return False