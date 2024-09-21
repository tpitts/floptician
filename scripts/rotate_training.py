import os
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Define input and output directories
INPUT_IMAGES_DIR = Path(r"C:\Users\tompi\projects\floptician\resources\truth\images\test")
INPUT_LABELS_DIR = Path(r"C:\Users\tompi\projects\floptician\resources\truth\labels\test")

OUTPUT_IMAGES_DIR = Path(r"C:\Users\tompi\projects\floptician\resources\truth\rotated_images\test")
OUTPUT_LABELS_DIR = Path(r"C:\Users\tompi\projects\floptician\resources\truth\rotated_labels\test")
VISUALIZATIONS_DIR = Path(r"C:\Users\tompi\projects\floptician\resources\truth\visualizations\test")

# Create output directories if they don't exist
for directory in [OUTPUT_IMAGES_DIR, OUTPUT_LABELS_DIR, VISUALIZATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Define the class names (update this list based on your actual classes)
CLASS_NAMES = [
    '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s', '4c', '4d', '4h', '4s',
    '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s', '7c', '7d', '7h', '7s',
    '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s', 'Ac', 'Ad', 'Ah', 'As',
    'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks', 'Qc', 'Qd', 'Qh', 'Qs',
    'Tc', 'Td', 'Th', 'Ts'
]  # Example classes

def rotate_image(image_path, angle=180):
    """Rotate an image by the specified angle."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Unable to read image {image_path}")
        return None
    # Rotate the image 180 degrees
    if angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 270:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # For angles not multiple of 90, use cv2.getRotationMatrix2D
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image

def adjust_label(label_path):
    """
    Adjust YOLO labels for a 180-degree rotation.
    YOLO format: class_id x_center y_center width height (all normalized)
    After 180-degree rotation:
        x_center_new = 1 - x_center
        y_center_new = 1 - y_center
    """
    if not label_path.exists():
        print(f"Warning: Label file {label_path} does not exist.")
        return None

    adjusted_labels = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Warning: Invalid label format in {label_path}: {line}")
                continue
            class_id, x_center, y_center, width, height = parts
            try:
                x_center = 1.0 - float(x_center)
                y_center = 1.0 - float(y_center)
                adjusted_label = f"{class_id} {x_center:.6f} {y_center:.6f} {width} {height}"
                adjusted_labels.append(adjusted_label)
            except ValueError:
                print(f"Warning: Non-float values in {label_path}: {line}")
                continue
    return adjusted_labels

def save_image(image, output_path):
    """Save the rotated image to the specified path."""
    cv2.imwrite(str(output_path), image)

def save_labels(adjusted_labels, output_path):
    """Save the adjusted labels to the specified path."""
    with open(output_path, 'w') as file:
        for label in adjusted_labels:
            file.write(label + '\n')

def draw_bounding_boxes(image, labels, output_path):
    """Draw bounding boxes on the image and save it."""
    # Convert image to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    width, height = pil_image.size

    try:
        # Attempt to load a TTF font
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        # If the font is not found, use the default font
        font = ImageFont.load_default()

    for label in labels:
        parts = label.strip().split()
        if len(parts) != 5:
            continue
        class_id, x_center, y_center, bbox_width, bbox_height = parts
        try:
            class_id = int(class_id)
            x_center = float(x_center) * width
            y_center = float(y_center) * height
            bbox_width = float(bbox_width) * width
            bbox_height = float(bbox_height) * height

            # Calculate top-left and bottom-right coordinates
            x1 = x_center - (bbox_width / 2)
            y1 = y_center - (bbox_height / 2)
            x2 = x_center + (bbox_width / 2)
            y2 = y_center + (bbox_height / 2)

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            # Draw label
            label_text = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
            
            # Get text size using draw.textbbox
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Draw filled rectangle for text background
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="red")
            
            # Draw text
            draw.text((x1 + 2, y1 - text_height - 2), label_text, fill="white", font=font)
        except (ValueError, IndexError):
            continue

    # Save the image with bounding boxes
    pil_image.save(output_path)

def process_dataset():
    """Process all images and labels by rotating them and adjusting labels."""
    image_files = list(INPUT_IMAGES_DIR.glob("*"))
    total_images = len(image_files)
    print(f"Found {total_images} images to process.")

    for idx, image_path in enumerate(image_files, 1):
        if image_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            print(f"Skipping non-image file: {image_path.name}")
            continue

        # Define corresponding label path
        label_path = INPUT_LABELS_DIR / (image_path.stem + ".txt")

        # Rotate image
        rotated_image = rotate_image(image_path, angle=180)
        if rotated_image is None:
            continue

        # Adjust labels
        adjusted_labels = adjust_label(label_path)
        if adjusted_labels is None:
            adjusted_labels = []

        # Define rotated filenames by appending 'rotated_'
        rotated_image_name = f"rotated_{image_path.name}"
        rotated_label_name = f"rotated_{image_path.stem}.txt"
        rotated_visualization_name = f"rotated_{image_path.name}"

        # Save rotated image
        output_image_path = OUTPUT_IMAGES_DIR / rotated_image_name
        save_image(rotated_image, output_image_path)

        # Save adjusted labels
        output_label_path = OUTPUT_LABELS_DIR / rotated_label_name
        save_labels(adjusted_labels, output_label_path)

        # Generate visualization with bounding boxes
        visualization_path = VISUALIZATIONS_DIR / rotated_visualization_name
        draw_bounding_boxes(rotated_image, adjusted_labels, visualization_path)

        print(f"Processed {idx}/{total_images}: {image_path.name}")

    print("Processing complete!")

if __name__ == "__main__":
    process_dataset()
