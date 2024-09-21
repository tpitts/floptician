import os
import cv2
from ultralytics import FastSAM
from datetime import datetime
import argparse
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_detected_objects(image_path):
    try:
        # Define the source as the input image path
        source = image_path

        # Create a FastSAM model
        model_path = "C:/Users/tompi/projects/floptician/resources/models/FastSAM-x.pt"
        model = FastSAM(model_path)
        logging.info(f"Loaded FastSAM model: {model_path}")

        # Run inference on the image
        everything_results = model(source, device="cpu", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        logging.info(f"Completed inference on image: {image_path}")

        # Load the image
        image = cv2.imread(image_path)

        # Get the current timestamp for unique folder naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_folder = f'C:/Users/tompi/projects/floptician/resources/masks/{timestamp}'

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        logging.info(f"Created output folder: {output_folder}")

        # Save bounding boxes
        boxes = everything_results[0].boxes.xyxy.cpu().numpy()  # Extract bounding boxes
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])

            # Crop the detected object from the image
            cropped_object = image[y1:y2, x1:x2]

            # Save the cropped object image
            output_path = os.path.join(output_folder, f'object_box_{idx + 1}.jpg')
            cv2.imwrite(output_path, cropped_object)
            logging.info(f'Saved: {output_path}')

        # Save masks
        if hasattr(everything_results[0], 'masks') and everything_results[0].masks is not None:
            masks = everything_results[0].masks.xy  # Extract masks
            logging.info(f"Number of masks: {len(masks)}")
            
            for idx, mask in enumerate(masks):
                logging.info(f"Processing mask {idx + 1}")
                
                # Create an empty mask image
                mask_img = np.zeros(image.shape[:2], dtype=np.uint8)

                # Convert mask to numpy array if it's not already
                mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask

                # Ensure mask coordinates are within image bounds
                mask_np = np.clip(mask_np, 0, [image.shape[1]-1, image.shape[0]-1])

                # Convert to integer coordinates
                mask_np = mask_np.astype(np.int32)

                # Draw filled polygon
                cv2.fillPoly(mask_img, [mask_np], 255)

                # Apply the mask to the original image
                masked_image = cv2.bitwise_and(image, image, mask=mask_img)

                # Save the masked image
                mask_path = os.path.join(output_folder, f'mask_{idx + 1}.png')
                cv2.imwrite(mask_path, masked_image)
                logging.info(f'Saved: {mask_path}')

                # Save the binary mask for debugging
                binary_mask_path = os.path.join(output_folder, f'binary_mask_{idx + 1}.png')
                cv2.imwrite(binary_mask_path, mask_img)
                logging.info(f'Saved binary mask: {binary_mask_path}')

        logging.info(f'All detected objects have been saved in {output_folder}')

    except Exception as e:
        logging.error(f"An error occurred: {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Detect and save objects in an image.')
    parser.add_argument('image_path', type=str, help='Path to the input image.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the function with the provided image path
    save_detected_objects(args.image_path)

if __name__ == '__main__':
    main()