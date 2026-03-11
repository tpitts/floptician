import os
import cv2
import numpy as np
from ultralytics import FastSAM
import logging
import argparse
from pathlib import Path
from datetime import datetime
from _common import resources_path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_images(input_folder, output_folder=None):
    try:
        # Create a FastSAM model
        model_path = resources_path("models", "FastSAM-x.pt")
        model = FastSAM(str(model_path))
        logging.info(f"Loaded FastSAM model: {model_path}")

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
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    logging.error(f"Failed to load image: {image_path}")
                    continue

                # Run inference on the image
                results = model(image_path, device="cpu", retina_masks=True, imgsz=1280, conf=0.4, iou=0.9)
                
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

                            # Create a 4-channel image (RGBA)
                            if image.shape[2] == 3:
                                rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
                            else:
                                rgba = image

                            # Set alpha channel based on the mask
                            rgba[:, :, 3] = mask_np

                            # Crop the image to the bounding box of the mask
                            y, x = np.where(mask_np > 0)
                            if len(y) > 0 and len(x) > 0:
                                top, bottom, left, right = y.min(), y.max(), x.min(), x.max()
                                cropped = rgba[top:bottom+1, left:right+1]

                                # Save the cropped image with transparent background
                                output_path = output_folder / f'{image_file[:-4]}_segment_{idx + 1}.png'
                                cv2.imwrite(str(output_path), cropped)
                                logging.info(f'Saved: {output_path}')
                            else:
                                logging.warning(f"Mask {idx + 1} for {image_file} is empty, skipping.")

                        except Exception as e:
                            logging.error(f"Error processing mask {idx + 1} for {image_file}: {e}")
                            continue
                else:
                    logging.warning(f"No masks found in the image: {image_file}")

        logging.info('All images have been processed.')

    except Exception as e:
        logging.error(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract segments from images using FastSAM.")
    parser.add_argument("input_dir", nargs="?", default=None, help="Path to the input directory containing images.")
    parser.add_argument("-o", "--output_dir", default=None, help="Path to the output directory. Defaults to resources/masks/<timestamp>.")
    args = parser.parse_args()

    input_folder = Path(args.input_dir) if args.input_dir else resources_path("video", "extract")
    output_folder = Path(args.output_dir) if args.output_dir else None
    process_images(input_folder, output_folder)

if __name__ == '__main__':
    main()
