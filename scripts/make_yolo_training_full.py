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
INPUT_SIZE = (1280, 720)
LETTERBOX_SIZE = (1280, 720)
OUTPUT_SIZE = (1280, 720)
CARD_SIZES = [63, 91, 120]  # Small, Medium, Large
BLUR_FACTORS = [0.0, 0.33, 0.67, 1.0]  # No blur, Light, Medium, Heavy
NUM_IMAGES = 3000
MAX_CARD_BACKS = 13
CARDS_PER_IMAGE = 13
MAX_OVERLAP = 0.03
TRAIN_SPLIT = 0.95

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
CARD_CLASSES = ['2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s', '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', 
                '6c', '6d', '6h', '6s', '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s', 
                'Ac', 'Ad', 'Ah', 'As', 'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks', 'Qc', 'Qd', 'Qh', 'Qs', 
                'Tc', 'Td', 'Th', 'Ts']

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
    """Generate a sequence of rotations with 0 and 180 degrees, plus random variations."""
    total_card_instances = total_images * cards_per_image
    
    # Create a list of base rotations (0 and 180 degrees)
    base_rotations = [0, 180]
    
    # Generate the sequence of rotations
    rotations = []
    for _ in range(total_card_instances):
        base_rotation = random.choice(base_rotations)
        variation = random.uniform(-20, 20)
        rotation = (base_rotation + variation) % 360
        rotations.append(round(rotation, 2))  # Round to 2 decimal places for practicality
    
    return rotations

def prepare_card_sequence_with_rotations(all_decks, total_images, cards_per_image):
    total_cards_needed = total_images * cards_per_image
    
    # Flatten the deck list and create a list of all cards
    all_cards = [card for deck in all_decks for card in deck]
    
    # Generate rotations
    rotations = generate_rotation_sequence(len(all_cards), total_images, cards_per_image)
    
    # Create card-rotation pairs
    card_rotation_pairs = []
    for _ in range(total_cards_needed):
        card = random.choice(all_cards)
        rotation = random.choice(rotations)
        card_rotation_pairs.append(CardWithRotation(card, rotation))
    
    # Shuffle the sequence
    random.shuffle(card_rotation_pairs)
    
    return card_rotation_pairs

def letterbox_image(image, target_size):
    """Resize image with unchanged aspect ratio using padding"""
    ih, iw = image.shape[:2]
    w, h = target_size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    # Determine the number of channels in the input image
    if image.shape[2] == 4:  # RGBA
        new_image = np.zeros((h,w,4), np.uint8)
        new_image[:,:,3] = 255  # Set alpha channel to fully opaque
    else:  # RGB
        new_image = np.zeros((h,w,3), np.uint8)
    
    new_image[:,:,:3] = 128  # fill with grey color

    image_resized = cv2.resize(image, (nw,nh))

    # Center the resized image in the new image
    top, bottom = (h-nh)//2, h-(h-nh)//2
    left, right = (w-nw)//2, w-(w-nw)//2
    
    if image.shape[2] == 4:  # RGBA
        new_image[top:bottom, left:right, :] = image_resized
    else:  # RGB
        new_image[top:bottom, left:right, :3] = image_resized

    return new_image

def generate_yolo_annotation(card_name, bounding_box, image_size):
    class_id = CARD_CLASSES.index(card_name)
    x, y, w, h = bounding_box
    x_center = (x + w / 2) / image_size[0]
    y_center = (y + h / 2) / image_size[1]
    width = w / image_size[0]
    height = h / image_size[1]
    return f"{class_id} {x_center} {y_center} {width} {height}"


def apply_blur(image, blur_factor):
    """
    Apply a combined Gaussian and defocus blur to the image.
    
    Parameters:
    - image: The input image to be blurred (numpy array).
    - blur_factor: A float representing the intensity of the blur. 
                   0.0 for no blur, higher values for stronger blur.

    Returns:
    - Blurred image.
    """
    # Apply Gaussian blur
    if blur_factor > 0.0:
        image_blurred = cv2.GaussianBlur(image, (5, 5), blur_factor)
    else:
        image_blurred = image  # No blur applied if blur_factor is 0

    # Determine defocus kernel size based on blur factor
    if blur_factor > 0.0:
        kernel_size = int(blur_factor / 3)  # Scale kernel size with blur factor
        kernel_size = max(5, kernel_size)  # Ensure minimum size for defocus blur
        if kernel_size % 2 == 0:
            kernel_size += 1  # Kernel size must be odd

        # Create a circular defocus kernel
        kernel = np.zeros((kernel_size, kernel_size), np.float32)
        cv2.circle(kernel, (kernel_size // 2, kernel_size // 2), kernel_size // 2, 1, -1)
        kernel /= kernel.sum()

        # Apply defocus blur
        image_blurred = cv2.filter2D(image_blurred, -1, kernel)

    return image_blurred
def generate_image(backgrounds, cards_subset, card_backs, output_path, label_path, bbox_image_path, index, card_size, blur_factor):
    background = random.choice(backgrounds).copy()
    background = cv2.resize(background, INPUT_SIZE)
    
    # Place card backs if available
    if card_backs:
        num_card_backs = random.randint(1, MAX_CARD_BACKS)
        for _ in range(num_card_backs):
            card_back = resize_image(random.choice(card_backs), card_size)
            angle = random.uniform(0, 360)
            card_back_rotated = rotate_image(card_back, angle)
            position = random_position(INPUT_SIZE, card_back_rotated.shape[:2])
            background = overlay_image(background, card_back_rotated, position)
    
    # Place cards and create bounding boxes
    bounding_boxes = []
    
    for card_with_rotation in cards_subset:
        card_resized = resize_image(card_with_rotation.card['image'], card_size)
        card_rotated = rotate_image(card_resized, card_with_rotation.rotation)
        
        # Apply specified blur to the card
        card_blurred = apply_blur(card_rotated, blur_factor) 
        
        # Try to place the card without excessive overlap
        max_attempts = 100
        for _ in range(max_attempts):
            position = random_position(INPUT_SIZE, card_blurred.shape[:2])
            temp_bg = background.copy()
            temp_bg = overlay_image(temp_bg, card_blurred, position)
            
            bounding_box = find_card_bounding_box(temp_bg, card_blurred, position)
            if bounding_box is None:
                continue
            
            overlaps = [check_overlap(bounding_box, existing_box) for _, existing_box in bounding_boxes]
            if sum(overlaps) <= 1 and (not overlaps or max(overlaps) <= MAX_OVERLAP):
                background = temp_bg
                bounding_boxes.append((card_with_rotation.card['name'], bounding_box))
                break
        else:
            print(f"Warning: Could not place card {card_with_rotation.card['name']} without excessive overlap after {max_attempts} attempts.")
    
    # Apply letterboxing
    letterboxed_image = letterbox_image(background, LETTERBOX_SIZE)
    
    # Resize to final output size
    final_image = cv2.resize(letterboxed_image, OUTPUT_SIZE)
    
    # Save clean image
    cv2.imwrite(str(output_path), final_image)
    
    # Adjust bounding boxes for letterboxing and resizing
    scale_x = OUTPUT_SIZE[0] / LETTERBOX_SIZE[0]
    scale_y = OUTPUT_SIZE[1] / LETTERBOX_SIZE[1]
    offset_x = (LETTERBOX_SIZE[0] - INPUT_SIZE[0]) / 2 * scale_x
    offset_y = (LETTERBOX_SIZE[1] - INPUT_SIZE[1]) / 2 * scale_y
    
    adjusted_bounding_boxes = []
    for card_name, (x, y, w, h) in bounding_boxes:
        new_x = x * scale_x + offset_x
        new_y = y * scale_y + offset_y
        new_w = w * scale_x
        new_h = h * scale_y
        adjusted_bounding_boxes.append((card_name, (new_x, new_y, new_w, new_h)))
    
    # Generate YOLO annotation
    with open(label_path, 'w') as f:
        for card_name, bbox in adjusted_bounding_boxes:
            yolo_annotation = generate_yolo_annotation(card_name, bbox, OUTPUT_SIZE)
            f.write(yolo_annotation + '\n')
    
    # Draw bounding boxes and save
    img_pil = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    for name, (x, y, w, h) in adjusted_bounding_boxes:
        draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
        draw.text((x, y-15), name, fill="red")
    
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
    
    total_combinations = len(CARD_SIZES) * len(BLUR_FACTORS)
    images_per_combination = NUM_IMAGES // total_combinations
    
    # Calculate the number of validation images
    num_val_images = int(NUM_IMAGES * (1 - TRAIN_SPLIT))
    
    # Generate all image metadata first
    all_images = []
    for card_size in CARD_SIZES:
        for blur_factor in BLUR_FACTORS:
            print(f"Preparing metadata for card size: {card_size}, blur factor: {blur_factor}")
            
            card_sequence = prepare_card_sequence_with_rotations(all_decks, images_per_combination, CARDS_PER_IMAGE)
            
            for i in range(images_per_combination):
                all_images.append((card_size, blur_factor, card_sequence[i*CARDS_PER_IMAGE:(i+1)*CARDS_PER_IMAGE]))

    # Randomly select validation images
    val_indices = set(random.sample(range(len(all_images)), num_val_images))

    # Generate images
    for image_counter, (card_size, blur_factor, cards_for_image) in enumerate(all_images):
        image_name = f"image_{image_counter:04d}.png"
        label_name = f"image_{image_counter:04d}.txt"
        bbox_image_name = f"bbox_image_{image_counter:04d}.png"
        
        if image_counter in val_indices:
            output_path = VAL_IMAGES_DIR / image_name
            label_path = VAL_LABELS_DIR / label_name
        else:
            output_path = TRAIN_IMAGES_DIR / image_name
            label_path = TRAIN_LABELS_DIR / label_name
        
        bbox_image_path = BBOX_IMAGES_DIR / bbox_image_name
        
        generate_image(backgrounds, cards_for_image, all_card_backs, output_path, label_path, bbox_image_path, image_counter, card_size, blur_factor)
        print(f"Generated image {image_counter+1}/{NUM_IMAGES}")

    # Create data.yaml
    create_data_yaml(DATASET_DIR / 'data.yaml', CARD_CLASSES)

if __name__ == "__main__":
    main()