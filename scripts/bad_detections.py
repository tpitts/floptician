import os
import sys
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import yaml
from ultralytics import YOLO

def create_unique_output_dir(base_dir='../output'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_dir = Path(base_dir) / f'bad_detections_{timestamp}'
    unique_dir.mkdir(parents=True, exist_ok=True)
    return unique_dir

def draw_boxes(img, boxes, color, class_names, label_prefix=''):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cls_id = int(box[5])
        score = box[4]
        label = f"{label_prefix}{class_names[cls_id]} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    # Paths to your model and data
    model_path = r'C:\Users\tompi\projects\moneta\runs\detect\train43\weights\best.pt'
    data_yaml = r'C:\Users\tompi\projects\floptician\resources\truth\test.yaml'

    # Create output directory
    output_dir = create_unique_output_dir()
    print(f"\nBad detections will be saved to: {output_dir.resolve()}\n")

    # Load the YOLO model
    print("Loading YOLO model...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    print("Model loaded successfully.\n")

    # Define the minimum confidence threshold for predictions
    min_confidence = 0.5  # Adjust this value as needed

    # Load the test dataset images
    print("Loading test dataset...")
    try:
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading data YAML file: {e}")
        sys.exit(1)

    # Ensure 'path' and 'test' keys exist in the data YAML
    if 'path' not in data or 'test' not in data:
        print("Error: 'path' and 'test' must be specified in the data YAML file.")
        sys.exit(1)

    test_images_path = data['path']
    test_set = data['test']

    # Handle both relative and absolute paths
    if not os.path.isabs(test_images_path):
        test_images_path = os.path.abspath(test_images_path)

    test_images_full_path = os.path.join(test_images_path, test_set)

    # Determine if 'test' is a file or directory
    test_images = []
    if os.path.isfile(test_images_full_path):
        # If 'test' is a file containing a list of image paths
        try:
            with open(test_images_full_path, 'r') as f:
                test_images = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading test images list from file: {e}")
            sys.exit(1)
    elif os.path.isdir(test_images_full_path):
        # If 'test' is a directory, list all image files in it
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        test_images = [str(p) for p in Path(test_images_full_path).glob('*') if p.suffix.lower() in supported_formats]
        if not test_images:
            print(f"No images found in directory: {test_images_full_path}")
            sys.exit(1)
    else:
        print(f"Test path '{test_images_full_path}' is neither a file nor a directory.")
        sys.exit(1)

    if not test_images:
        print("No test images to process.")
        sys.exit(1)

    print(f"Total test images found: {len(test_images)}\n")

    # Map class indices to class names
    class_names = model.model.names

    # Initialize counters for summary
    total_images = len(test_images)
    total_false_positives = 0
    total_false_negatives = 0

    print("Processing images individually...\n")

    # Iterate over each image
    for idx, img_path in enumerate(test_images, 1):
        img_name = Path(img_path).name
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error reading image: {img_path}")
            continue

        # Run prediction on the image
        try:
            result = model.predict(
                source=img_path,
                conf=0.001,
                iou=0.5,
                save=False,
                verbose=False
            )[0]
        except Exception as e:
            print(f"Error during model prediction for image {img_path}: {e}")
            continue

        # Get predicted boxes
        pred_boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = box.conf.cpu().numpy()[0]
            cls_id = int(box.cls.cpu().numpy()[0])

            if score >= min_confidence:
                pred_boxes.append([x1, y1, x2, y2, score, cls_id])
        pred_boxes = np.array(pred_boxes)

        # Get ground truth labels from label files
        label_path = Path(img_path).with_suffix('.txt')
        # Adjust path to point to the 'labels' directory
        if 'images' in label_path.parts:
            parts = list(label_path.parts)
            idx_images = parts.index('images')
            parts[idx_images] = 'labels'
            label_path = Path(*parts)
        else:
            print(f"Could not locate 'images' in path for image: {img_path}")
            continue

        gt_array = []
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts_line = line.strip().split()
                        if len(parts_line) >= 5:
                            cls_id = int(parts_line[0])
                            x_center, y_center, width, height = map(float, parts_line[1:5])
                            # Convert from YOLO format to bounding box coordinates
                            img_height, img_width = img.shape[:2]
                            x1 = int((x_center - width / 2) * img_width)
                            y1 = int((y_center - height / 2) * img_height)
                            x2 = int((x_center + width / 2) * img_width)
                            y2 = int((y_center + height / 2) * img_height)
                            gt_array.append([x1, y1, x2, y2, 1.0, cls_id])  # Confidence 1.0
                        else:
                            print(f"Invalid label format in file {label_path}, line: {line.strip()}")
                gt_array = np.array(gt_array)
            except Exception as e:
                print(f"Error reading label file {label_path}: {e}")
                continue
        else:
            print(f"No label file found for image: {img_path}")
            continue

        # Compute IoU between predictions and ground truths
        iou_threshold = 0.5
        false_positives = []
        false_negatives = []

        matched_gt_indices = set()
        matched_pred_indices = set()

        for pred_idx, pred_box in enumerate(pred_boxes):
            max_iou = 0
            max_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_array):
                # Compute IoU
                xi1 = max(pred_box[0], gt_box[0])
                yi1 = max(pred_box[1], gt_box[1])
                xi2 = min(pred_box[2], gt_box[2])
                yi2 = min(pred_box[3], gt_box[3])
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

                pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

                union_area = pred_area + gt_area - inter_area

                iou = inter_area / union_area if union_area > 0 else 0

                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx

            if max_iou >= iou_threshold and pred_box[5] == gt_array[max_gt_idx][5]:
                # Correct detection
                matched_gt_indices.add(max_gt_idx)
                matched_pred_indices.add(pred_idx)
            else:
                # False positive
                false_positives.append(pred_box)

        # False negatives
        for gt_idx, gt_box in enumerate(gt_array):
            if gt_idx not in matched_gt_indices:
                false_negatives.append(gt_box)

        # Update counters
        total_false_positives += len(false_positives)
        total_false_negatives += len(false_negatives)

        # Log per-image results
        print(f"Processing image {idx}/{total_images}: {img_name}")
        print(f"  False Positives: {len(false_positives)}")
        print(f"  False Negatives: {len(false_negatives)}")

        # If there are bad detections, visualize them
        if false_positives or false_negatives:
            # Draw false positives in red
            if len(false_positives) > 0:
                draw_boxes(img, false_positives, color=(0, 0, 255), class_names=class_names, label_prefix='FP: ')

            # Draw false negatives in blue
            if len(false_negatives) > 0:
                draw_boxes(img, false_negatives, color=(255, 0, 0), class_names=class_names, label_prefix='FN: ')

            # Optionally, draw matched detections in green
            # matched_boxes = [pred_boxes[i] for i in matched_pred_indices]
            # draw_boxes(img, matched_boxes, color=(0, 255, 0), class_names=class_names, label_prefix='TP: ')

            # Save the image
            output_path = output_dir / img_name
            try:
                cv2.imwrite(str(output_path), img)
                print(f"  Saved bad detection visualization: {output_path}\n")
            except Exception as e:
                print(f"  Error saving image {output_path}: {e}\n")
        else:
            print(f"  No bad detections for this image.\n")

    # Output summary
    print("Processing completed.\n")
    print("Summary:")
    print(f"  Total images processed: {total_images}")
    print(f"  Total false positives: {total_false_positives}")
    print(f"  Total false negatives: {total_false_negatives}")
    print(f"  Bad detections saved in: {output_dir.resolve()}")

if __name__ == '__main__':
    main()