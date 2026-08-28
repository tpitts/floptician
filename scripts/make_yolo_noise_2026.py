import argparse
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from _common import output_path, resources_path

NUM_WORKERS = max(1, os.cpu_count() or 1)

# Constants
INPUT_SIZE = (1280, 720)
LETTERBOX_SIZE = (1280, 720)
OUTPUT_SIZE = (1280, 720)
CARD_SIZES = [63, 91, 120]  # Small, Medium, Large
BLUR_FACTORS = [0.0, 0.15, 0.23, 0.3]  # No blur, Light, Medium, Heavy
NUM_IMAGES = 2000
NUM_VAL_IMAGES = 100
MAX_CARD_BACKS = 17
CARDS_PER_IMAGE = 13
MAX_OVERLAP = 0.03

# Over-expressed ranks: face cards and Ace of Spades get 2x weight
OVEREXPRESS_CARDS = {'Jc', 'Jd', 'Jh', 'Js', 'Qc', 'Qd', 'Qh', 'Qs',
                     'Kc', 'Kd', 'Kh', 'Ks', 'As'}

# Define a named tuple to hold card and rotation information
CardWithRotation = namedtuple('CardWithRotation', ['card', 'rotation'])

# Input Paths
BACKGROUND_DIR = resources_path("training_background")
DECKS_DIR = resources_path("decks")
TRAINING_BG_DIR = resources_path("training_background")
TRUTH_DIR = resources_path("truth-v1")

# Function to create a unique output directory and return run_id
def create_unique_output_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_dir = resources_path("training", timestamp)
    unique_dir.mkdir(parents=True, exist_ok=True)
    return unique_dir, timestamp

# Create unique output directory and get run_id
OUTPUT_DIR, RUN_ID = create_unique_output_dir()

# Output Paths
DATASET_DIR = OUTPUT_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
TRAIN_IMAGES_DIR = IMAGES_DIR / "train"
VAL_IMAGES_DIR = IMAGES_DIR / "val"
TRAIN_LABELS_DIR = LABELS_DIR / "train"
VAL_LABELS_DIR = LABELS_DIR / "val"
TEST_IMAGES_DIR = IMAGES_DIR / "test"
TEST_LABELS_DIR = LABELS_DIR / "test"
BBOX_IMAGES_DIR = OUTPUT_DIR / "bbox_images"

# Ensure output directories exist
for dir in [TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_LABELS_DIR,
            TEST_IMAGES_DIR, TEST_LABELS_DIR, BBOX_IMAGES_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

# --- Fixed CARD_CLASSES Setup ---
CARD_CLASSES = [
    '2c', '2d', '2h', '2s',
    '3c', '3d', '3h', '3s',
    '4c', '4d', '4h', '4s',
    '5c', '5d', '5h', '5s',
    '6c', '6d', '6h', '6s',
    '7c', '7d', '7h', '7s',
    '8c', '8d', '8h', '8s',
    '9c', '9d', '9h', '9s',
    'Ac', 'Ad', 'Ah', 'As',
    'Jc', 'Jd', 'Jh', 'Js',
    'Kc', 'Kd', 'Kh', 'Ks',
    'Qc', 'Qd', 'Qh', 'Qs',
    'Tc', 'Td', 'Th', 'Ts'
]
CARD_CLASS_INDEX = {name: i for i, name in enumerate(CARD_CLASSES)}

def load_images(directory):
    images = []
    for file in directory.glob("*"):
        if file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            image = cv2.imread(str(file))
            if image is not None:
                images.append(image)
    return images

def load_card_backs(decks_dir):
    """Load card back images from the decks-new root directory."""
    backs = []
    i = 1
    while True:
        back_file_png = decks_dir / f"back{i}.png"
        back_file_jpg = decks_dir / f"back{i}.jpg"

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
        print(f"Warning: No card backs found in {decks_dir}")
    return backs

def load_cards(deck_dir):
    """Load card images directly from a deck directory (no 'cards' subfolder)."""
    cards = []
    for card_file in deck_dir.glob("*"):
        if card_file.suffix.lower() in ['.png', '.jpg', '.jpeg'] and not card_file.stem.startswith("back"):
            card_name = card_file.stem
            if card_name in CARD_CLASSES:
                card_image = cv2.imread(str(card_file), cv2.IMREAD_UNCHANGED)
                if card_image is not None:
                    cards.append({"name": card_name, "image": card_image})
    return cards

def resize_image(image, width):
    aspect_ratio = image.shape[1] / image.shape[0]
    height = int(width / aspect_ratio)
    return cv2.resize(image, (width, height))

def rotate_image(image, angle):
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    height, width = image.shape[:2]
    diagonal = int(math.ceil(math.sqrt(height**2 + width**2)))

    background = np.zeros((diagonal, diagonal, 4), dtype=np.uint8)
    x_offset = (diagonal - width) // 2
    y_offset = (diagonal - height) // 2
    background[y_offset:y_offset+height, x_offset:x_offset+width] = image

    major_angle = 90 * round(angle / 90)
    if major_angle != 0:
        background = np.rot90(background, k=int(major_angle // 90))

    minor_angle = angle - major_angle
    if minor_angle != 0:
        rotation_matrix = cv2.getRotationMatrix2D((diagonal//2, diagonal//2), minor_angle, 1.0)
        background = cv2.warpAffine(background, rotation_matrix, (diagonal, diagonal),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))

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

    if x < 0:
        overlay = overlay[:, -x:]
        w = overlay.shape[1]
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        h = overlay.shape[0]
        y = 0
    if x + w > background.shape[1]:
        overlay = overlay[:, :background.shape[1]-x]
        w = background.shape[1]-x
    if y + h > background.shape[0]:
        overlay = overlay[:background.shape[0]-y, :]
        h = background.shape[0]-y

    if background.shape[2] == 3:
        background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

    if overlay.shape[2] == 3:
        alpha_channel = np.ones((overlay.shape[0], overlay.shape[1]), dtype=np.uint8) * 255
        overlay = np.dstack([overlay, alpha_channel])

    roi = background[y:y+h, x:x+w]

    ov_alpha = overlay[:, :, 3] / 255.0
    bg_alpha = roi[:, :, 3] / 255.0

    out_alpha = ov_alpha + bg_alpha * (1 - ov_alpha)
    out_alpha = np.where(out_alpha == 0, 1, out_alpha)

    ov_alpha_3d = ov_alpha[:, :, np.newaxis]
    bg_alpha_3d = bg_alpha[:, :, np.newaxis]
    out_alpha_3d = out_alpha[:, :, np.newaxis]
    roi[:, :, :3] = ((overlay[:, :, :3] * ov_alpha_3d + roi[:, :, :3] * bg_alpha_3d * (1 - ov_alpha_3d)) / out_alpha_3d).astype(np.uint8)

    roi[:, :, 3] = (out_alpha * 255).astype(np.uint8)
    background[y:y+h, x:x+w] = roi

    return background

def find_card_bounding_box(background, card_image, position):
    x, y = position
    card_h, card_w = card_image.shape[:2]
    bg_h, bg_w = background.shape[:2]

    if background.shape[2] == 4:
        card_gray = cv2.cvtColor(card_image, cv2.COLOR_BGRA2GRAY)
    else:
        card_gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)

    _, card_mask = cv2.threshold(card_gray, 1, 255, cv2.THRESH_BINARY)
    full_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)

    y_start = max(0, y)
    y_end = min(bg_h, y + card_h)
    x_start = max(0, x)
    x_end = min(bg_w, x + card_w)

    card_y_start = max(0, -y)
    card_x_start = max(0, -x)

    full_mask[y_start:y_end, x_start:x_end] = card_mask[card_y_start:card_y_start+(y_end-y_start),
                                                        card_x_start:card_x_start+(x_end-x_start)]

    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return (x, y, w, h)
    else:
        return None

def check_overlap(box1, box2, max_overlap=0.2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = overlap_x * overlap_y

    area1 = w1 * h1
    area2 = w2 * h2

    overlap_ratio1 = overlap_area / area1 if area1 > 0 else 0
    overlap_ratio2 = overlap_area / area2 if area2 > 0 else 0

    return max(overlap_ratio1, overlap_ratio2) > max_overlap

def generate_rotation_sequence(num_cards, total_images, cards_per_image):
    """Generate a sequence of rotations with 0 and 180 degrees, plus random variations."""
    total_card_instances = total_images * cards_per_image

    base_rotations = [0, 180]
    rotations = []
    for _ in range(total_card_instances):
        base_rotation = random.choice(base_rotations)
        variation = random.uniform(-20, 20)
        rotation = (base_rotation + variation) % 360
        rotations.append(round(rotation, 2))

    return rotations

def build_weighted_card_pool(all_decks):
    """Build a weighted card pool where over-expressed cards appear 2x."""
    pool = []
    for deck in all_decks:
        for card in deck:
            pool.append(card)
            if card['name'] in OVEREXPRESS_CARDS:
                pool.append(card)  # Add a second copy for 2x weight
    return pool

def prepare_card_sequence_with_rotations(weighted_pool, total_images, cards_per_image):
    total_cards_needed = total_images * cards_per_image

    rotations = generate_rotation_sequence(len(weighted_pool), total_images, cards_per_image)

    card_rotation_pairs = []
    for _ in range(total_cards_needed):
        card = random.choice(weighted_pool)
        rotation = random.choice(rotations)
        card_rotation_pairs.append(CardWithRotation(card, rotation))

    random.shuffle(card_rotation_pairs)
    return card_rotation_pairs

def letterbox_image(image, target_size):
    """Resize image with unchanged aspect ratio using padding"""
    ih, iw = image.shape[:2]
    w, h = target_size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    if image.shape[2] == 4:
        new_image = np.zeros((h,w,4), np.uint8)
        new_image[:,:,3] = 255
    else:
        new_image = np.zeros((h,w,3), np.uint8)

    new_image[:,:,:3] = 128

    image_resized = cv2.resize(image, (nw,nh))

    top, bottom = (h-nh)//2, h-(h-nh)//2
    left, right = (w-nw)//2, w-(w-nw)//2

    if image.shape[2] == 4:
        new_image[top:bottom, left:right, :] = image_resized
    else:
        new_image[top:bottom, left:right, :3] = image_resized

    return new_image

def generate_yolo_annotation(card_name, bounding_box, image_size):
    class_id = CARD_CLASS_INDEX[card_name]
    x, y, w, h = bounding_box
    x_center = (x + w / 2) / image_size[0]
    y_center = (y + h / 2) / image_size[1]
    width = w / image_size[0]
    height = h / image_size[1]
    return f"{class_id} {x_center} {y_center} {width} {height}"

def apply_blur(image, blur_factor):
    if blur_factor > 0.0:
        image_blurred = cv2.GaussianBlur(image, (5, 5), blur_factor)
    else:
        image_blurred = image

    if blur_factor > 0.0:
        kernel_size = int(blur_factor / 3)
        kernel_size = max(5, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.zeros((kernel_size, kernel_size), np.float32)
        cv2.circle(kernel, (kernel_size // 2, kernel_size // 2), kernel_size // 2, 1, -1)
        kernel /= kernel.sum()

        image_blurred = cv2.filter2D(image_blurred, -1, kernel)

    return image_blurred

def apply_noise(image, gaussian_sigma_range=(15, 45), shadow_intensity_range=(0.1, 0.25), shadow_angle_range=(0, 360),
               glare_probability=0.3, glare_intensity_range=(0.03, 0.08), glare_size_range=(0.05, 0.10), glare_angle_range=(0, 360)):
    """Apply Gaussian noise, gradient shadows, and glare effects to non-transparent areas."""
    if image.shape[2] == 4:
        mask = image[:, :, 3] > 0
        mask_3ch = np.stack([mask]*3, axis=2)
    else:
        mask_3ch = np.ones_like(image[:, :, :3], dtype=bool)

    noisy_image = image.copy()

    # 1. Gaussian Noise
    gaussian_sigma = random.uniform(*gaussian_sigma_range)
    gaussian_noise = np.random.normal(0, gaussian_sigma, noisy_image[:, :, :3].shape).astype(np.float32)

    noisy_image[:, :, :3] = np.where(
        mask_3ch,
        np.clip(noisy_image[:, :, :3].astype(np.float32) + gaussian_noise, 0, 255).astype(np.uint8),
        noisy_image[:, :, :3]
    )

    # 2. Gradient Shadows
    shadow_intensity = random.uniform(*shadow_intensity_range)
    shadow_angle = random.uniform(*shadow_angle_range)

    height, width = noisy_image.shape[:2]
    angle_rad = np.deg2rad(shadow_angle)
    X, Y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    gradient = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
    gradient = 1 - gradient
    gradient = np.stack([gradient]*3, axis=2)
    shadow = (gradient * 255 * shadow_intensity).astype(np.uint8)

    noisy_image[:, :, :3] = np.where(
        mask_3ch,
        cv2.addWeighted(noisy_image[:, :, :3], 1.0, shadow, 1.0, 0),
        noisy_image[:, :, :3]
    )

    # 3. Glare
    if random.random() < glare_probability:
        glare_intensity = random.uniform(*glare_intensity_range)
        glare_size = random.uniform(*glare_size_range)

        glare_radius = int(glare_size * width)
        glare_overlay = np.zeros((height, width, 4), dtype=np.uint8)
        center_x = random.randint(int(0.1 * width), int(0.9 * width))
        center_y = random.randint(int(0.1 * height), int(0.9 * height))

        num_flares = random.randint(1, 3)
        for _ in range(num_flares):
            flare_radius = int(glare_radius * random.uniform(0.5, 1.0))
            flare_color = (255, 255, 255, int(255 * glare_intensity))
            cv2.circle(glare_overlay, (center_x, center_y), flare_radius, flare_color, -1)

        alpha_glare = glare_overlay[:, :, 3] / 255.0
        alpha_image = noisy_image[:, :, 3] / 255.0 if noisy_image.shape[2] == 4 else np.ones((height, width))

        alpha_glare_3d = alpha_glare[:, :, np.newaxis]
        alpha_image_3d = alpha_image[:, :, np.newaxis]
        noisy_image[:, :, :3] = (noisy_image[:, :, :3] * (1 - alpha_glare_3d) +
                                 glare_overlay[:, :, :3] * alpha_glare_3d * alpha_image_3d).astype(np.uint8)

        if noisy_image.shape[2] == 4:
            noisy_image[:, :, 3] = (alpha_image * 255).astype(np.uint8)

    return noisy_image

def generate_image(backgrounds, cards_subset, card_backs, output_dir, label_dir, bbox_dir, index, card_size, blur_factor, run_id):
    background = random.choice(backgrounds).copy()
    background = cv2.resize(background, INPUT_SIZE)

    # Place card backs if available
    if card_backs:
        num_card_backs = random.randint(1, MAX_CARD_BACKS)
        for _ in range(num_card_backs):
            card_back = resize_image(random.choice(card_backs), card_size)
            angle = random.uniform(0, 360)
            card_back_rotated = rotate_image(card_back, angle)

            card_back_blurred = apply_blur(card_back_rotated, blur_factor)
            card_back_noisy = apply_noise(
                card_back_blurred,
                gaussian_sigma_range=(5, 15),
                shadow_intensity_range=(0.3, 0.7),
                shadow_angle_range=(0, 360),
                glare_probability=0.3,
                glare_intensity_range=(0.2, 0.8),
                glare_size_range=(0.05, 0.15),
                glare_angle_range=(0, 360)
            )

            position = random_position(INPUT_SIZE, card_back_noisy.shape[:2])
            background = overlay_image(background, card_back_noisy, position)

    # Place cards and create bounding boxes
    bounding_boxes = []

    for card_with_rotation in cards_subset:
        card_resized = resize_image(card_with_rotation.card['image'], card_size)
        card_rotated = rotate_image(card_resized, card_with_rotation.rotation)

        card_blurred = apply_blur(card_rotated, blur_factor)
        card_noisy = apply_noise(
            card_blurred,
            gaussian_sigma_range=(5, 11),
            shadow_intensity_range=(0.1, 0.3),
            shadow_angle_range=(0, 360),
            glare_probability=0.4,
            glare_intensity_range=(0.05, 0.28),
            glare_size_range=(0.05, 0.15),
            glare_angle_range=(0, 360)
        )

        max_attempts = 100
        for _ in range(max_attempts):
            position = random_position(INPUT_SIZE, card_noisy.shape[:2])
            temp_bg = background.copy()
            temp_bg = overlay_image(temp_bg, card_noisy, position)

            bounding_box = find_card_bounding_box(temp_bg, card_noisy, position)
            if bounding_box is None:
                continue

            overlaps = [check_overlap(bounding_box, existing_box) for _, existing_box in bounding_boxes]
            if sum(overlaps) <= 1 and (not overlaps or max(overlaps) <= MAX_OVERLAP):
                background = temp_bg
                bounding_boxes.append((card_with_rotation.card['name'], bounding_box))
                break
        else:
            print(f"Warning: Could not place card {card_with_rotation.card['name']} without excessive overlap after {max_attempts} attempts.")

    letterboxed_image = letterbox_image(background, LETTERBOX_SIZE)
    final_image = cv2.resize(letterboxed_image, OUTPUT_SIZE)

    unique_image_name = f"image_{run_id}_{index:04d}.png"
    cv2.imwrite(str(output_dir / unique_image_name), final_image)

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

    label_lines = []
    for card_name, bbox in adjusted_bounding_boxes:
        yolo_annotation = generate_yolo_annotation(card_name, bbox, OUTPUT_SIZE)
        label_lines.append(yolo_annotation)

    unique_label_name = f"image_{run_id}_{index:04d}.txt"
    with open(label_dir / unique_label_name, 'w') as f:
        f.write('\n'.join(label_lines) + '\n')

    img_pil = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    for name, (x, y, w, h) in adjusted_bounding_boxes:
        draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
        draw.text((x, y-15), name, fill="red")

    unique_bbox_image_name = f"bbox_image_{run_id}_{index:04d}.png"
    img_pil.save(bbox_dir / unique_bbox_image_name)

def copy_truth_data(truth_dir, test_images_dir, test_labels_dir):
    """Copy truth/test images and labels into the dataset."""
    src_images = truth_dir / "images" / "test"
    src_labels = truth_dir / "labels" / "test"

    copied = 0
    if src_images.exists():
        for img_file in src_images.glob("*"):
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                shutil.copy2(str(img_file), str(test_images_dir / img_file.name))
                # Copy matching label if it exists
                label_file = src_labels / (img_file.stem + ".txt")
                if label_file.exists():
                    shutil.copy2(str(label_file), str(test_labels_dir / label_file.name))
                copied += 1
    print(f"Copied {copied} truth test images/labels")
    return copied

def copy_training_backgrounds(bg_dir, train_images_dir, train_labels_dir, run_id):
    """Copy training background images as negative examples (empty label files)."""
    copied = 0
    if bg_dir.exists():
        for img_file in bg_dir.glob("*"):
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                dest_name = f"bg_{run_id}_{copied:04d}{img_file.suffix}"
                shutil.copy2(str(img_file), str(train_images_dir / dest_name))
                # Create empty label file
                label_path = train_labels_dir / f"bg_{run_id}_{copied:04d}.txt"
                label_path.touch()
                copied += 1
    print(f"Copied {copied} training background images as negative examples")
    return copied

def create_data_yaml(yaml_path, dataset_dir, class_names):
    names_dict = {i: name for i, name in enumerate(class_names)}
    data = {
        'path': str(dataset_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': names_dict
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def main(num_images=NUM_IMAGES, num_val_images=NUM_VAL_IMAGES):
    backgrounds = load_images(BACKGROUND_DIR)
    if not backgrounds:
        print("Error: No background images found.")
        return

    all_decks = []
    for deck_dir in DECKS_DIR.iterdir():
        if deck_dir.is_dir():
            deck_cards = load_cards(deck_dir)
            if deck_cards:
                all_decks.append(deck_cards)
                print(f"Loaded {len(deck_cards)} cards from {deck_dir.name}")

    if not all_decks:
        print("Error: No card images found in any deck.")
        return

    # Load card backs from decks-new root
    all_card_backs = load_card_backs(DECKS_DIR)
    print(f"Loaded {len(all_card_backs)} card backs")

    # Build weighted card pool (face cards + As at 2x)
    weighted_pool = build_weighted_card_pool(all_decks)
    total_cards = sum(len(d) for d in all_decks)
    print(f"Card pool: {total_cards} unique cards -> {len(weighted_pool)} weighted entries "
          f"({len(OVEREXPRESS_CARDS)} card types at 2x)")

    total_combinations = len(CARD_SIZES) * len(BLUR_FACTORS)
    images_per_combination = num_images // total_combinations

    # Generate all image metadata first
    all_images = []
    for card_size in CARD_SIZES:
        for blur_factor in BLUR_FACTORS:
            print(f"Preparing metadata for card size: {card_size}, blur factor: {blur_factor}")

            card_sequence = prepare_card_sequence_with_rotations(weighted_pool, images_per_combination, CARDS_PER_IMAGE)

            for i in range(images_per_combination):
                all_images.append((card_size, blur_factor, card_sequence[i*CARDS_PER_IMAGE:(i+1)*CARDS_PER_IMAGE]))

    # Randomly select validation images
    val_indices = set(random.sample(range(len(all_images)), num_val_images))

    # Generate images using thread pool
    total = len(all_images)
    print(f"Generating {total} images using {NUM_WORKERS} workers...")

    def _gen(image_counter, card_size, blur_factor, cards_for_image):
        if image_counter in val_indices:
            out_dir = VAL_IMAGES_DIR
            lbl_dir = VAL_LABELS_DIR
        else:
            out_dir = TRAIN_IMAGES_DIR
            lbl_dir = TRAIN_LABELS_DIR

        generate_image(
            backgrounds,
            cards_for_image,
            all_card_backs,
            out_dir,
            lbl_dir,
            BBOX_IMAGES_DIR,
            image_counter,
            card_size,
            blur_factor,
            RUN_ID
        )

    completed = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(_gen, i, cs, bf, ci): i
            for i, (cs, bf, ci) in enumerate(all_images)
        }
        for future in as_completed(futures):
            future.result()  # propagate exceptions
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"Generated {completed}/{total} images")

    # Copy truth test data
    copy_truth_data(TRUTH_DIR, TEST_IMAGES_DIR, TEST_LABELS_DIR)

    # Copy training backgrounds as negative examples
    copy_training_backgrounds(TRAINING_BG_DIR, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, RUN_ID)

    # Create data.yaml
    create_data_yaml(DATASET_DIR / 'data.yaml', DATASET_DIR, CARD_CLASSES)

    num_train = len(all_images) - num_val_images
    print(f"\nDone! {len(all_images)} synthetic images generated ({num_train} train / {num_val_images} val)")
    print(f"Output: {OUTPUT_DIR}")
    print(f"\nReady to train:")
    print(f'yolo task=detect mode=train model=yolov8l.pt data="{DATASET_DIR / "data.yaml"}" '
          f'epochs=80 imgsz=1280 degrees=11 translate=0.015 shear=0.035 perspective=0.0003 '
          f'flipud=0 fliplr=0 mixup=0 copy_paste=0 patience=20 mosaic=0 scale=0.3 batch=-1 hsv_h=0.045')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic YOLO training data from resources/decks.")
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES,
                        help=f"synthetic image budget, spread over card-size x blur combinations (default {NUM_IMAGES})")
    parser.add_argument("--num-val", type=int, default=NUM_VAL_IMAGES,
                        help=f"how many of those images go to the val split (default {NUM_VAL_IMAGES})")
    args = parser.parse_args()
    main(num_images=args.num_images, num_val_images=args.num_val)
