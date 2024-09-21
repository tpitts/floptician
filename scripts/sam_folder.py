import os
import cv2
import numpy as np
from ultralytics import SAM
import logging
from pathlib import Path
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_images(input_folder):
    try:
        # Create a SAM2 model
        model_path = "C:/Users/tompi/projects/floptician/resources/models/sam2_b.pt"
        model = SAM(model_path)
        logging.info(f"Loaded SAM2 model: {model_path}")

        # Create output folder inside the input directory
        output_folder = Path(os.path.join(input_folder, 'output'))
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

                            # Create a 4-channel image (RGBA)
                            rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

                            # Set alpha channel based on the mask
                            rgba[:, :, 3] = mask_np

                            # Save the image with transparent background
                            output_path = output_folder / f'{Path(image_file).stem}_mask_{idx + 1}.png'
                            cv2.imwrite(str(output_path), rgba)
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
    parser = argparse.ArgumentParser(description="Process images and save mask portions.")
    parser.add_argument('input_folder', help='Path to the input folder containing images.')
    args = parser.parse_args()

    process_images(args.input_folder)

if __name__ == '__main__':
    main()
