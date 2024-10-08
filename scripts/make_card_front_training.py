import os
import random
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import shutil
import yaml
from datetime import datetime
from collections import namedtuple

# Constants
OUTPUT_SIZE = (1280, 720)
CARD_WIDTH_RANGE = (80, 120)
NUM_IMAGES = 420
MAX_CARD_BACKS = 17
CARDS_PER_IMAGE = 13
MAX_OVERLAP = 0.2
TRAIN_SPLIT = 0.9
BLUR_CARDS = True
BLUR_MIN = 0.1
BLUR_MAX = 1.5

# Define a named tuple to hold card and rotation information
CardWithRotation = namedtuple('CardWithRotation', ['card', 'rotation'])

# Input Paths
BACKGROUND_DIR = Path(r"..\resources\background")
DECKS_DIR = Path(r"..\resources\decks")

# Function to create a unique output directory
def create_unique_output_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_dir = Path(r"..\output") / f"run_{timestamp}"
    unique_dir.mkdir(parents=True, exist_ok=True)
    return unique_dir

# Create unique output directory
OUTPUT_DIR = create_unique_output_dir()

# Output Paths
DATASET_DIR = OUTPUT_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
TRAIN_IMAGES_DIR = IMAGES_DIR / "train"
VAL_IMAGES_DIR = IMAGES_DIR / "val"
TRAIN_LABELS_DIR = LABELS_DIR / "train"
VAL_LABELS_DIR = LABELS_DIR / "val"
BBOX_IMAGES_DIR = OUTPUT_DIR / "bbox_images"

# Ensure output directories exist
for dir in [TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_LABELS_DIR, BBOX_IMAGES_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

# Define CARD_CLASSES globally
CARD_CLASSES = ['card_front']

def load_images(directory):
    images = []
    for file in directory.glob("*"):
        if file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            image = cv2.imread(str(file))
            if image is not None:
                images.append(image)
    return images

def load_card_backs(cards_dir):
    backs = []
    i = 1
    while True:
        back_file_png = cards_dir / f"back{i}.png"
        back_file_jpg = cards_dir / f"back{i}.jpg"
        
        if back_file_png.exists():
            back_image = cv2.imread(str(back_file_png), cv2.IMREAD_UNCHANGED)
            if back_image is not None:
                backs.append(back_image)
            i += 1
        elif back_file_jpg.exists():
            back_image = cv2.imread(str(back_file_jpg), cv2.IMREAD_UNCHANGED)
            if back_image is not None:
                backs.append(back_image)
            i += 1
        else:
            break

    if not backs:
        print(f"Warning: No card backs found in {cards_dir}")
    return backs

def load_cards(cards_dir):
    cards = []
    for card_file in cards_dir.glob("*"):
        if card_file.suffix.lower() in ['.png', '.jpg', '.jpeg'] and not card_file.stem.startswith("back"):
            card_name = card_file.stem
            card_image = cv2.imread(str(card_file), cv2.IMREAD_UNCHANGED)
            if card_image is not None:
                cards.append({"name": card_name, "image": card_image})
    return cards

def resize_image(image, width):
    aspect_ratio = image.shape[1] / image.shape[0]
    height = int(width / aspect_ratio)
    return cv2.resize(image, (width, height))

def rotate_image(image, angle):
    # Convert to RGBA if it's not already
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # Get the image size
    height, width = image.shape[:2]
    
    # Calculate the size of the new image to contain the entire rotated image
    diagonal = int(math.ceil(math.sqrt(height**2 + width**2)))
    
    # Create a transparent background
    background = np.zeros((diagonal, diagonal, 4), dtype=np.uint8)
    
    # Calculate the position to paste the original image
    x_offset = (diagonal - width) // 2
    y_offset = (diagonal - height) // 2
    background[y_offset:y_offset+height, x_offset:x_offset+width] = image
    
    # Rotate the image in two steps
    # Step 1: rotate by a multiple of 90 degrees
    major_angle = 90 * round(angle / 90)
    if major_angle != 0:
        background = np.rot90(background, k=int(major_angle // 90))
    
    # Step 2: rotate by the remaining angle
    minor_angle = angle - major_angle
    if minor_angle != 0:
        rotation_matrix = cv2.getRotationMatrix2D((diagonal//2, diagonal//2), minor_angle, 1.0)
        background = cv2.warpAffine(background, rotation_matrix, (diagonal, diagonal), 
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=(0, 0, 0, 0))
    
    # Crop the image to remove unnecessary transparent borders
    alpha_channel = background[:,:,3]
    coords = cv2.findNonZero(alpha_channel)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = background[y:y+h, x:x+w]
    else:
        cropped = background
    
    return cropped


def apply_random_blur(image):
    if not BLUR_CARDS:
        return image
    blur_amount = random.uniform(BLUR_MIN, BLUR_MAX)
    return cv2.GaussianBlur(image, (5, 5), blur_amount)

def random_position(image_size, card_size):
    return (
        random.randint(0, image_size[0] - card_size[0]),
        random.randint(0, image_size[1] - card_size[1])
    )

def overlay_image(background, overlay, position):
    x, y = position
    h, w = overlay.shape[:2]
    
    # Ensure the overlay fits within the background
    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > background.shape[1]: w = background.shape[1] - x
    if y + h > background.shape[0]: h = background.shape[0] - y
    
    # If the background is BGR, convert it to BGRA
    if background.shape[2] == 3:
        background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
    
    # Blend the overlay onto the background
    alpha_overlay = overlay[:h, :w, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay
    
    for c in range(3):  # Blend RGB channels
        background[y:y+h, x:x+w, c] = (alpha_overlay * overlay[:h, :w, c] + 
                                       alpha_background * background[y:y+h, x:x+w, c])
    
    # Update alpha channel
    background[y:y+h, x:x+w, 3] = (alpha_overlay * overlay[:h, :w, 3] + 
                                   alpha_background * background[y:y+h, x:x+w, 3])
    
    return background

def find_card_bounding_box(background, card_image, position):
    x, y = position
    card_h, card_w = card_image.shape[:2]
    bg_h, bg_w = background.shape[:2]
    
    # Create a mask of the card
    card_gray = cv2.cvtColor(card_image, cv2.COLOR_BGRA2GRAY)
    _, card_mask = cv2.threshold(card_gray, 1, 255, cv2.THRESH_BINARY)
    
    # Create a full-size mask
    full_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    
    # Calculate the region where the card should be placed
    y_start = max(0, y)
    y_end = min(bg_h, y + card_h)
    x_start = max(0, x)
    x_end = min(bg_w, x + card_w)
    
    # Calculate the corresponding region in the card mask
    card_y_start = max(0, -y)
    card_x_start = max(0, -x)
    
    # Place the visible part of the card mask onto the full mask
    full_mask[y_start:y_end, x_start:x_end] = card_mask[card_y_start:card_y_start+(y_end-y_start), 
                                                        card_x_start:card_x_start+(x_end-x_start)]
    
    # Find contours of the card in its new position
    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contours[0])
        return (x, y, w, h)
    else:
        return None

def check_overlap(box1, box2, max_overlap=0.2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the overlapping area
    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = overlap_x * overlap_y

    # Calculate the area of both boxes
    area1 = w1 * h1
    area2 = w2 * h2

    # Check if the overlap percentage exceeds the maximum allowed for either box
    overlap_ratio1 = overlap_area / area1 if area1 > 0 else 0
    overlap_ratio2 = overlap_area / area2 if area2 > 0 else 0
    
    return max(overlap_ratio1, overlap_ratio2) > max_overlap

def generate_rotation_sequence(num_cards, total_images, cards_per_image):
    """Generate a sequence of rotations including the base sequence and additional rotations."""
    base_sequence = [0, 2, 180, 182]
    
    total_card_instances = total_images * cards_per_image
    min_rotations_per_card = math.ceil(total_card_instances / num_cards / 2)  # Ensure at least 2 appearances
    
    num_additional_rotations = max(0, min_rotations_per_card - len(base_sequence))
    
    if num_additional_rotations > 0:
        step = 360 / num_additional_rotations
        additional_rotations = [round(i * step) % 360 for i in range(num_additional_rotations)]
    else:
        additional_rotations = []
    
    rotations = base_sequence + additional_rotations
    return rotations  # Keeping all duplicates

def prepare_card_sequence_with_rotations(all_decks, total_images, cards_per_image):
    """Prepare a sequence of cards with assigned rotations, ensuring each card has every rotation at least twice."""
    num_cards = sum(len(deck) for deck in all_decks)
    rotations = generate_rotation_sequence(num_cards, total_images, cards_per_image)
    
    # Create all possible card-rotation pairs
    all_card_rotation_pairs = []
    for deck in all_decks:
        for card in deck:
            for rotation in rotations:
                all_card_rotation_pairs.append(CardWithRotation(card, rotation))
    
    # Calculate how many times we need to repeat the sequence
    total_cards_needed = total_images * cards_per_image
    repeats = math.ceil(total_cards_needed / len(all_card_rotation_pairs))
    
    # Create the full sequence by repeating the pairs
    card_sequence = all_card_rotation_pairs * repeats
    
    # Cut off any excess cards
    card_sequence = card_sequence[:total_cards_needed]
    
    # Shuffle the sequence
    random.shuffle(card_sequence)
    
    return card_sequence

def generate_yolo_annotation(bounding_box, image_size):
    x, y, w, h = bounding_box
    x_center = (x + w / 2) / image_size[0]
    y_center = (y + h / 2) / image_size[1]
    width = w / image_size[0]
    height = h / image_size[1]
    return f"0 {x_center} {y_center} {width} {height}"  # 0 is the class ID for 'card_front'

def generate_image(backgrounds, cards_subset, card_backs, output_path, label_path, bbox_image_path, index):
    background = random.choice(backgrounds).copy()
    background = cv2.resize(background, OUTPUT_SIZE)
    
    card_width = random.randint(*CARD_WIDTH_RANGE)
    
    # Place card backs if available
    if card_backs:
        num_card_backs = random.randint(1, MAX_CARD_BACKS)
        for _ in range(num_card_backs):
            card_back = resize_image(random.choice(card_backs), card_width)
            angle = random.uniform(0, 360)
            card_back_rotated = rotate_image(card_back, angle)
            position = random_position(OUTPUT_SIZE, card_back_rotated.shape[:2])
            background = overlay_image(background, card_back_rotated, position)
    
    # Place cards and create bounding boxes
    bounding_boxes = []
    
    for card_with_rotation in cards_subset:
        card_resized = resize_image(card_with_rotation.card['image'], card_width)
        card_rotated = rotate_image(card_resized, card_with_rotation.rotation)
        
        # Apply random blur to the card
        card_blurred = apply_random_blur(card_rotated)
        
        # Try to place the card without excessive overlap
        max_attempts = 100
        for _ in range(max_attempts):
            position = random_position(OUTPUT_SIZE, card_blurred.shape[:2])
            temp_bg = background.copy()
            temp_bg = overlay_image(temp_bg, card_blurred, position)
            
            bounding_box = find_card_bounding_box(temp_bg, card_blurred, position)
            if bounding_box is None:
                continue
            
            overlaps = [check_overlap(bounding_box, existing_box) for existing_box in bounding_boxes]
            if sum(overlaps) <= 1 and (not overlaps or max(overlaps) <= MAX_OVERLAP):
                background = temp_bg
                bounding_boxes.append(bounding_box)
                break
        else:
            print(f"Warning: Could not place a card without excessive overlap after {max_attempts} attempts.")
    
    # Save clean image
    cv2.imwrite(str(output_path), cv2.cvtColor(background, cv2.COLOR_BGRA2BGR))
    
    # Generate YOLO annotation
    with open(label_path, 'w') as f:
        for bbox in bounding_boxes:
            yolo_annotation = generate_yolo_annotation(bbox, OUTPUT_SIZE)
            f.write(yolo_annotation + '\n')
    
    # Draw bounding boxes and save
    img_pil = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGRA2RGBA))
    draw = ImageDraw.Draw(img_pil)
    
    for x, y, w, h in bounding_boxes:
        draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
        draw.text((x, y-15), "card_front", fill="red")
    
    img_pil.save(bbox_image_path)

def create_data_yaml(output_path, class_names):
    data = {
        'train': '../dataset/images/train',
        'val': '../dataset/images/val',
        'nc': len(class_names),
        'names': class_names
    }
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def main():
    backgrounds = load_images(BACKGROUND_DIR)
    if not backgrounds:
        print("Error: No background images found.")
        return

    all_decks = []
    all_card_backs = []
    
    for deck_dir in DECKS_DIR.iterdir():
        if deck_dir.is_dir():
            cards_dir = deck_dir / "cards"
            if cards_dir.exists():
                deck_cards = load_cards(cards_dir)
                if deck_cards:
                    all_decks.append(deck_cards)
                    
                    # Load card backs for this deck
                    deck_backs = load_card_backs(cards_dir)
                    all_card_backs.extend(deck_backs)
    
    if not all_decks:
        print("Error: No card images found in any deck.")
        return
    
    if not all_card_backs:
        print("Warning: No card back images found in any deck. Proceeding without card backs.")
    
    card_sequence = prepare_card_sequence_with_rotations(all_decks, NUM_IMAGES, CARDS_PER_IMAGE)
    
    for i in range(NUM_IMAGES):
        image_name = f"image_{i:04d}.png"
        label_name = f"image_{i:04d}.txt"
        bbox_image_name = f"bbox_image_{i:04d}.png"
        
        output_path = TRAIN_IMAGES_DIR / image_name if i < NUM_IMAGES * TRAIN_SPLIT else VAL_IMAGES_DIR / image_name
        label_path = TRAIN_LABELS_DIR / label_name if i < NUM_IMAGES * TRAIN_SPLIT else VAL_LABELS_DIR / label_name
        bbox_image_path = BBOX_IMAGES_DIR / bbox_image_name
        
        # Get the next batch of cards from the sequence
        start_index = i * CARDS_PER_IMAGE
        end_index = start_index + CARDS_PER_IMAGE
        cards_for_image = card_sequence[start_index:end_index]
        
        generate_image(backgrounds, cards_for_image, all_card_backs, output_path, label_path, bbox_image_path, i)
        print(f"Generated image {i+1}/{NUM_IMAGES}")
    
    # Create data.yaml
    create_data_yaml(DATASET_DIR / 'data.yaml', CARD_CLASSES)

if __name__ == "__main__":
    main()