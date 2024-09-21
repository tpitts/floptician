import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path

def straighten_and_crop_card(image_path, output_path):
    # Open image with Pillow to preserve transparency
    pil_image = Image.open(image_path)
    
    # Convert to numpy array for OpenCV processing
    np_image = np.array(pil_image)
    
    # Split the alpha channel
    rgb = np_image[:,:,:3]
    alpha = np_image[:,:,3]
    
    # Create a binary mask from the alpha channel
    _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (assuming it's the card)
        card_contour = max(contours, key=cv2.contourArea)
        
        # Get the rotated rectangle that bounds the contour
        rect = cv2.minAreaRect(card_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Get the angle of rotation
        angle = rect[2]
        if abs(angle) < 45:
            angle = angle
        elif angle < 0:
            angle = angle + 90
        else:
            angle = angle - 90
        
        # Rotate the image
        (h, w) = np_image.shape[:2]
        center = rect[0]
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(np_image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        
        # Get the new bounding rectangle
        rotated_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rotated_contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if rotated_contours:
            rotated_card_contour = max(rotated_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(rotated_card_contour)
            
            # Crop the image
            cropped = rotated[y:y+h, x:x+w]
            
            # Convert back to PIL Image
            straightened = Image.fromarray(cropped)
            
            # Save the straightened and cropped image
            straightened.save(output_path)
            print(f"Saved straightened and cropped image: {output_path}")
        else:
            print(f"No contour found after rotation in {image_path}")
    else:
        print(f"No contours found in {image_path}")

def process_directory(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for img_path in input_path.glob('*.png'):
        output_file = output_path / img_path.name
        straighten_and_crop_card(str(img_path), str(output_file))

if __name__ == "__main__":
    input_directory = "C:/Users/tompi/projects/floptician/resources/masks/cards"
    output_directory = "C:/Users/tompi/projects/floptician/resources/masks/straight_cards"
    process_directory(input_directory, output_directory)