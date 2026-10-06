# -*- coding: utf-8 -*-
"""
  Author : Shoujun Huang, Junjie Liu, Shousheng Luo, Huafeng Xie, Jing Yuan, Dexing Kong, Jianfeng Zhang
  Usage  : python sourceCode.py
  Reference  : Our Paper in Biomedical Signal Processing and Control:
    From automatic detection to segmentation: A label-efficient SAM pipeline with feature-guided star-shape priors for breast ultrasound 
"""

import numpy as np
import cv2
import json
from segment_anything import sam_model_registry, SamPredictor
import os
import random
import logging
# Configure logging for debug messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Get current script directory
current_dir = os.path.dirname(os.path.realpath(__file__))
# Load bounding‑box data from CV_bbox.json
json_path = os.path.join(current_dir, "CV_bbox.json")
with open(json_path, 'r', encoding='utf-8') as json_file:
    bounding_boxes = json.load(json_file)
if __name__ == '__main__':
    sam_checkpoint = "models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    # Image folder path
    image_folder = os.path.join(current_dir, "mixdata1")
    # Load SAM model and move to target device
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    logging.info("SAM model loaded successfully")
    # Initialize SamPredictor
    predictor = SamPredictor(sam)
    # Create or open log file for skipped images
    skipped_txt_path = os.path.join(current_dir, "skipped_images.txt")
    skipped_file = open(skipped_txt_path, 'a', encoding='utf-8')
    # Iterate over each box entry in JSON
    for i, box_data in enumerate(bounding_boxes):
        file_name = box_data.get('file_name', 'unknown_filename')
        logging.info(f"Processing image {i+1}: {file_name}")
        # Check existence of 'new_bbox' field
        new_bbox = box_data.get('new_bbox')
        if new_bbox is None:
            logging.warning(f"Image {file_name} missing 'new_bbox' field, skipped.")
            skipped_file.write(f"{file_name} missing 'new_bbox'\n")
            continue
        # Build absolute image path and read
        image_path = os.path.join(image_folder, file_name)
        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Cannot read image: {image_path}")
            skipped_file.write(f"{file_name} read failed\n")
            continue
        # Generate base name for output files
        cropped_image = box_data.get('cropped_image', file_name)
        if isinstance(cropped_image, (list, tuple)):
            base_name = '_'.join(map(str, cropped_image))
            if not base_name.strip('_'):
                base_name = os.path.splitext(os.path.basename(file_name))[0]
                logging.warning(f"cropped_image yields empty string, fallback to file_name as basename.")
        elif isinstance(cropped_image, str):
            base_name = os.path.splitext(os.path.basename(cropped_image))[0]
        else:
            base_name = os.path.splitext(os.path.basename(file_name))[0]
            logging.warning(f"cropped_image format invalid, fallback to file_name as basename.")
        # Preserve '_re' suffix logic
        if '_re' not in base_name:
            name_parts = os.path.splitext(os.path.basename(file_name))[0].split('_')
            if len(name_parts) > 1 and name_parts[-1].startswith('re'):
                base_name = '_'.join(name_parts)
            else:
                logging.warning(f"basename '{base_name}' lacks '_re', keep original.")
        # Output paths for visualization and mask
        combined_filename = f"{base_name}_combined_results.png"
        combined_path = os.path.join(os.path.dirname(image_path), combined_filename)
        mask_path = os.path.join(os.path.dirname(image_path), f"{base_name}_mask.png")
        # Skip if result already exists
        if os.path.exists(combined_path):
            logging.info(f"Result already exists: {combined_filename}, skip.")
            continue
        # BGR → RGB conversion
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Draw bounding box on image
        image_with_box = original_image.copy()
        cv2.rectangle(
            image_with_box,
            (int(new_bbox[0]), int(new_bbox[1])),
            (int(new_bbox[0] + new_bbox[2]), int(new_bbox[1] + new_bbox[3])),
            (255, 0, 0),
            2
        )
        # Generate Otsu binarized mask for random foreground/background point sampling
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        _, otsu_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Sample points inside prompt bbox
        x_min = int(new_bbox[0])
        y_min = int(new_bbox[1])
        w_box = int(new_bbox[2])
        h_box = int(new_bbox[3])
        x_max = x_min + w_box
        y_max = y_min + h_box
        # Local Otsu mask within bbox region (255=background, 0=foreground)
        bbox_otsu_mask = otsu_mask[y_min:y_max, x_min:x_max]
        # Randomly pick background point inside bbox by Otsu
        background_points_rel = np.argwhere(bbox_otsu_mask == 255)
        if len(background_points_rel) == 0:
            logging.warning(f"No background point inside prompt box, skip: {file_name}")
            skipped_file.write(f"{file_name} no background point inside box\n")
            continue
        selected_background_point_rel = random.choice(background_points_rel)
        bg_point_y = y_min + selected_background_point_rel[0]
        bg_point_x = x_min + selected_background_point_rel[1]
        # Randomly pick foreground point inside bbox by Otsu
        foreground_points_rel = np.argwhere(bbox_otsu_mask == 0)
        if len(foreground_points_rel) == 0:
            logging.warning(f"No foreground point inside prompt box, skip: {file_name}")
            skipped_file.write(f"{file_name} no foreground point inside box\n")
            continue
        selected_foreground_point_rel = random.choice(foreground_points_rel)
        fg_point_y = y_min + selected_foreground_point_rel[0]
        fg_point_x = x_min + selected_foreground_point_rel[1]
        logging.info(f"Sampled background point: ({bg_point_x}, {bg_point_y}), foreground point: ({fg_point_x}, {fg_point_y})")
        # Draw sampled points (red=foreground, green=background)
        image_with_points = original_image.copy()
        cv2.circle(image_with_points, (bg_point_x, bg_point_y), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.circle(image_with_points, (fg_point_x, fg_point_y), radius=5, color=(255, 0, 0), thickness=-1)
        # Feed image into SAM predictor
        predictor.set_image(original_image)
        # Construct input box format [x_min, y_min, x_max, y_max]
        input_box = np.array([x_min, y_min, x_max, y_max])
        # SAM inference: bbox prompt + foreground‑background point prompts
        point_coords_bg_fg = np.array([[bg_point_x, bg_point_y], [fg_point_x, fg_point_y]])
        point_labels_bg_fg = np.array([0, 1])  # 0=background, 1=foreground
        masks_bg_fg, _, _ = predictor.predict(
            point_coords=point_coords_bg_fg,
            point_labels=point_labels_bg_fg,
            box=input_box[None, :],
            multimask_output=False,
        )
        mask_bg_fg = (masks_bg_fg[0] * 255).astype(np.uint8)
        # Draw segmentation contour on original image
        contours_bg_fg, _ = cv2.findContours(mask_bg_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmented_image_bg_fg = original_image.copy()
        cv2.drawContours(segmented_image_bg_fg, contours_bg_fg, -1, (0, 0, 255), 2)
        # Save segmentation mask
        cv2.imwrite(mask_path, mask_bg_fg)
        logging.info(f"Saved BG+FG segmentation mask: {mask_path}")
        # Build concatenated visualization image
        try:
            # Concatenate three sub‑images: box‑overlaid, point‑overlaid, segmentation result
            images_to_concat = [image_with_box, image_with_points, segmented_image_bg_fg]
            height, width, _ = original_image.shape
            concat_images = []
            for img in images_to_concat:
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img_resized = cv2.resize(img, (width, height))
                concat_images.append(img_resized)
            combined_image = cv2.hconcat(concat_images)
            # Add text annotation on concatenated image
            cv2.putText(
                combined_image,
                f"Image: {os.path.basename(image_path)}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            cv2.putText(
                combined_image,
                f"Box: {new_bbox}",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            # Save visualization, convert RGB back to BGR for OpenCV imwrite
            cv2.imwrite(combined_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
            logging.info(f"Saved combined visualization: {combined_path}")
        except Exception as e:
            logging.error(f"Error when saving concatenated image: {e}")
            skipped_file.write(f"{file_name} failed to save combined image: {e}\n")
    skipped_file.close()
    logging.info("All images processed. Check skipped_images.txt for skipped entries.")
