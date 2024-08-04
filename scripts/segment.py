import torch
import numpy as np
import cv2
import os
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import math
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image(image_path, max_size=1024):
    """Load and preprocess an image from a file."""
    logging.info(f"Loading and preprocessing image from: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image from path: {image_path}")
    
    # Resize image if it's too large
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w*scale), int(h*scale)))
    
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logging.info("Image loaded and preprocessed successfully.")
    return image, image_rgb

def run_sam_on_image(image, checkpoint_path, device):
    """Run the Segment Anything model on the provided image."""
    model_type = "vit_h"  # Use "vit_h" for the huge model

    # Check if the model type is available in the registry
    if model_type not in sam_model_registry:
        raise KeyError(f"Model type '{model_type}' is not available in the registry.")

    # Initialize the model
    logging.info(f"Loading model: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    logging.info(f"Model loaded and moved to device: {device}")

    # Initialize the automatic mask generator with optimized settings
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    logging.info("Initialized SamAutomaticMaskGenerator with optimized settings.")

    # Generate masks with progress bar
    logging.info("Generating masks.")
    masks = mask_generator.generate(image)
    logging.info(f"Generated {len(masks)} masks.")
    
    # Check if any masks were generated
    if len(masks) == 0:
        logging.warning("No masks were generated. The output images may be blank.")
    else:
        # Log information about the first mask
        logging.info(f"First mask shape: {masks[0]['segmentation'].shape}")
        logging.info(f"First mask sum: {masks[0]['segmentation'].sum()}")
    
    return masks

def is_rectangle_like(mask, bbox, area_threshold=0.85, aspect_ratio_tolerance=0.2):
    """Check if a mask is approximately rectangular."""
    mask_area = np.sum(mask)
    bbox_area = bbox[2] * bbox[3]  # width * height
    area_ratio = mask_area / bbox_area
    
    aspect_ratio = bbox[2] / bbox[3]  # width / height
    common_ratios = [1, 4/3, 16/9]
    
    is_common_ratio = any(abs(aspect_ratio - ratio) / ratio < aspect_ratio_tolerance for ratio in common_ratios)
    
    logging.info(f"Area ratio: {area_ratio:.2f}, Aspect ratio: {aspect_ratio:.2f}")
    
    return area_ratio > area_threshold and is_common_ratio

def rotate_image(image, angle):
    """Rotate an image by a given angle using Pillow."""
    pil_image = Image.fromarray(image)
    rotated = pil_image.rotate(-angle, expand=True, resample=Image.BICUBIC)
    return np.array(rotated)

def process_mask(image, mask, i, output_path):
    """Process a single mask, detecting rectangle-like shapes and applying rotation if necessary."""
    # Get the binary mask and bounding box
    binary_mask = mask['segmentation'].astype(np.uint8)
    bbox = mask['bbox']

    # Crop the image to the bounding box
    x, y, w, h = bbox
    cropped_image = image[y:y+h, x:x+w]
    cropped_mask = binary_mask[y:y+h, x:x+w]

    if is_rectangle_like(cropped_mask, (x, y, w, h)):
        # Get the rotated rectangle
        contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(contours[0])
        angle = rect[2]
        
        # Adjust angle
        if angle < -45:
            angle = 90 + angle
        
        # Apply rotation if the angle is greater than the threshold
        if abs(angle) > 0.1:
            logging.info(f"Rotating mask {i+1} by {angle:.2f} degrees")
            rotated_image = rotate_image(cropped_image, angle)
            rotated_mask = rotate_image(cropped_mask, angle)
            
            # Re-crop to remove any black borders
            coords = cv2.findNonZero(rotated_mask)
            x, y, w, h = cv2.boundingRect(coords)
            rotated_image = rotated_image[y:y+h, x:x+w]
            rotated_mask = rotated_mask[y:y+h, x:x+w]
            
            # Create RGBA image
            rgba = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2RGBA)
            # Set alpha channel based on the rotated mask
            rgba[:, :, 3] = rotated_mask * 255
        else:
            logging.info(f"Mask {i+1} is already straight (angle: {angle:.2f} degrees)")
            # If no significant rotation, just use the original cropped image
            rgba = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2RGBA)
            rgba[:, :, 3] = cropped_mask * 255
    else:
        logging.info(f"Mask {i+1} is not rectangle-like, processing without rotation")
        # For non-rectangular shapes, process as before
        rgba = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = cropped_mask * 255

    # Save the result
    mask_output_path = os.path.join(output_path, f"mask_{i+1}.png")
    Image.fromarray(rgba).save(mask_output_path)

def save_masks(image, masks, output_path):
    """Save each identified mask as a separate cropped image file with original colors and transparent background."""
    logging.info(f"Saving masks to: {output_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, mask in tqdm(enumerate(masks), total=len(masks), desc="Saving masks"):
        process_mask(image, mask, i, output_path)

    logging.info(f"Saved {len(masks)} masks.")

def main(image_path, output_path, checkpoint_path):
    """Main function to load an image, run SAM, and save identified objects."""
    # Create a unique subfolder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_path = os.path.join(output_path, f"run_{timestamp}")
    os.makedirs(run_output_path, exist_ok=True)
    logging.info(f"Created output folder for this run: {run_output_path}")

    # Load image
    image, image_rgb = load_image(image_path)
    
    # Check if the image was loaded correctly
    if image.size == 0 or image_rgb.size == 0:
        raise ValueError("Failed to load the image. Please check the image path.")
    
    logging.info(f"Loaded image shape: {image.shape}")

    # Determine the device to use
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Run SAM on image
    masks = run_sam_on_image(image_rgb, checkpoint_path, device)

    # Save identified masks with original colors
    save_masks(image_rgb, masks, run_output_path)

    logging.info(f"Process completed. Results saved in: {run_output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scan an image file and parse out objects using the Segment Anything model.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument("output_path", type=str, help="Directory where the output mask images will be saved.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the SAM model checkpoint file.")
    args = parser.parse_args()

    # Add CUDA availability check
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logging.info(f"Device count: {torch.cuda.device_count()}")
        logging.info(f"Device name: {torch.cuda.get_device_name(0)}")

    main(args.image_path, args.output_path, args.checkpoint_path)