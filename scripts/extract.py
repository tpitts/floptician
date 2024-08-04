import cv2
import numpy as np
from PIL import Image
import os
import argparse
from datetime import datetime
import glob

def create_directory(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def detect_card_locations(image):
    """Detect card locations using a simpler approach."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    card_mask = cv2.bitwise_not(green_mask)
    contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def flood_fill_transparent(image, lower_green, upper_green, lo_diff, up_diff):
    """Turn green background transparent using flood fill."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    seed_points = [(1, 1), (1, w - 2), (h - 2, 1), (h - 2, w - 2)]

    for seed_point in seed_points:
        try:
            cv2.floodFill(image, flood_mask, (seed_point[1], seed_point[0]), (0, 0, 0, 0), lo_diff, up_diff, cv2.FLOODFILL_FIXED_RANGE)
        except cv2.error as e:
            print(f"Error with seed point {seed_point}: {e}")
    
    output_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    output_image[flood_mask[1:-1, 1:-1] == 1, 3] = 0
    
    return output_image

def process_and_save_cards(image, contours, output_path, image_name):
    """Process detected card locations and save the cards."""
    card_images = []
    for i, contour in enumerate(contours):
        contour_area = cv2.contourArea(contour)
        print(f"Contour {i} area: {contour_area}")
        if contour_area < 5000:
            print(f"Skipping contour {i} due to small area")
            continue
        
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.intp)
        
        width = int(rect[1][0])
        height = int(rect[1][1])
        
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(box.astype(np.float32), dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))
        
        if width > height:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            width, height = height, width
        
        card_images.append(warped)
    
    save_cards(card_images, output_path, image_name)

def save_cards(card_images, output_path, image_name):
    """Save processed cards to the output directory."""
    for i, card in enumerate(card_images):
        card_rgb = cv2.cvtColor(card, cv2.COLOR_BGRA2RGBA)
        card_pil = Image.fromarray(card_rgb)
        card_pil.save(os.path.join(output_path, f"{image_name}_card_{i+1}.png"))

def process_image(image_path, output_path):
    """Process a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Unable to read image at {image_path}")
        return

    print(f"Processing image: {image_path}")
    print(f"Image shape: {image.shape}")

    image_name = os.path.splitext(os.path.basename(image_path))[0]

    lower_green = np.array([30, 30, 30])
    upper_green = np.array([100, 255, 255])

    contours = detect_card_locations(image)
    print(f"Number of contours found: {len(contours)}")

    cv2.imwrite(os.path.join(output_path, f"{image_name}_original.png"), image)
    lo_diff = up_diff = (100, 100, 100)
    flooded_image = flood_fill_transparent(image.copy(), lower_green, upper_green, lo_diff, up_diff)
    flooded_image_path = os.path.join(output_path, f"{image_name}_flooded.png")
    cv2.imwrite(flooded_image_path, flooded_image)
    print(f"Flood-filled image saved to {flooded_image_path}")

    process_and_save_cards(flooded_image, contours, output_path, image_name)

def main():
    parser = argparse.ArgumentParser(description="Detect and process cards from images, removing green background.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing image files.")
    parser.add_argument("output_dir", type=str, help="Directory where the output card images will be saved.")
    
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"run_{timestamp}")
    create_directory(output_path)

    image_files = glob.glob(os.path.join(args.input_dir, '*.jpg')) + glob.glob(os.path.join(args.input_dir, '*.png'))
    
    for image_file in image_files:
        process_image(image_file, output_path)

    print(f"All images processed. Results saved in: {output_path}")

if __name__ == "__main__":
    main()