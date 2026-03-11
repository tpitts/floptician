import os
import cv2
import numpy as np
from ultralytics import SAM
import logging
import argparse
from pathlib import Path
from datetime import datetime
from _common import resources_path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_images(input_folder, output_folder=None):
    try:
        # Create a SAM2 model
        model_path = resources_path("models", "sam2_b.pt")
        model = SAM(str(model_path))
        logging.info(f"Loaded SAM2 model: {model_path}")

        # Create output folder with timestamp
        if output_folder is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_folder = resources_path("masks", timestamp)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created output folder: {output_folder}")

        # Process each image in the input folder
        for image_file in os.listdir(input_folder):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_folder, image_file)
                logging.info(f"Processing image: {image_path}")

                # Load the image
                image = cv2.imread(image_path)
                if image is None:
                    logging.error(f"Failed to load image: {image_path}")
                    continue

                # Run inference on the image using the everything prompt
                results = model(image_path)
                
                if results[0].masks is not None:
                    masks = results[0].masks.data
                    logging.info(f"Number of masks for {image_file}: {len(masks)}")
                    
                    for idx, mask in enumerate(masks):
                        try:
                            # Convert mask to numpy array and ensure it's binary
                            mask_np = mask.cpu().numpy()
                            mask_np = (mask_np > 0.5).astype(np.uint8) * 255

                            # Resize mask to match image dimensions
                            mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]))

                            # Find contours in the mask
                            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            # Filter contours based on area
                            min_area = 1000  # Adjust this value as needed
                            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

                            for contour_idx, contour in enumerate(valid_contours):
                                # Create a mask for this contour
                                contour_mask = np.zeros(mask_np.shape, dtype=np.uint8)
                                cv2.drawContours(contour_mask, [contour], 0, 255, -1)

                                # Create a 4-channel image (RGBA)
                                rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

                                # Set alpha channel based on the contour mask
                                rgba[:, :, 3] = contour_mask

                                # Get bounding box of the contour
                                x, y, w, h = cv2.boundingRect(contour)

                                # Crop the image to the bounding box
                                cropped = rgba[y:y+h, x:x+w]

                                # Save the cropped image with transparent background
                                output_path = output_folder / f'{image_file[:-4]}_segment_{idx + 1}_{contour_idx + 1}.png'
                                cv2.imwrite(str(output_path), cropped)
                                logging.info(f'Saved: {output_path}')

                        except Exception as e:
                            logging.error(f"Error processing mask {idx + 1} for {image_file}: {e}")
                            continue
                else:
                    logging.warning(f"No masks found in the image: {image_file}")

        logging.info('All images have been processed.')

    except Exception as e:
        logging.error(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract segments from images using SAM 2.")
    parser.add_argument("input_dir", nargs="?", default=None, help="Path to the input directory containing images.")
    parser.add_argument("-o", "--output_dir", default=None, help="Path to the output directory. Defaults to resources/masks/<timestamp>.")
    args = parser.parse_args()

    input_folder = Path(args.input_dir) if args.input_dir else resources_path("video", "extract")
    output_folder = Path(args.output_dir) if args.output_dir else None
    process_images(input_folder, output_folder)

if __name__ == '__main__':
    main()
