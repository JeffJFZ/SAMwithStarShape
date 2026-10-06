# -*- coding: utf-8 -*-
"""
  Author : Shoujun Huang, Junjie Liu, Shousheng Luo, Huafeng Xie, Jing Yuan, Dexing Kong, Jianfeng Zhang
  Usage  : python sourceCode.py
  Description  :
      1. Expand each bbox by 1/20 of height at top and bottom, 1/20 of width at left and right to form enlarged_bbox;
      2. Crop grayscale image only within enlarged_bbox and perform Chan‑Vese CV evolution;
      3. Map obtained new contour back to original image coordinates to generate new_bbox;
      4. Fall back to original bbox on failure (no contour, contour too small, etc.).
"""
import os
import cv2
import numpy as np
import json
import math
from scipy.ndimage import gaussian_filter
from collections import defaultdict
import logging
import traceback
# ------------------------------------------------------------
# === Path Configuration === 
# ------------------------------------------------------------
current_dir        = r"PATH\CV4"
image_folder       = r"PATH\BUSI_images"
json_path          = r"PATH\BUSI_yolov10.json"
cv_bbox_json_path  = r"PATH\BUSI_yolov10+CV.json"
# ------------------------------------------------------------
# --- Runtime Options ---
SKIP_EXISTING_NEW_BBOX = False      # True: skip bbox with existing processed result
PROCESS_ONLY_NO_CHANGE = False      # True: only rerun images recorded in no_change.txt
# === Global enlarge ratio adjustment ===
ENLARGE_RATIO = 0.05                   # expand 5% of original bbox on each side
# ============ Logging Configuration ============
failure_logger = logging.getLogger('FailureLogger')
failure_logger.setLevel(logging.WARNING)
failure_log_path = os.path.join(current_dir, 'defeat_picture.log')
failure_handler  = logging.FileHandler(failure_log_path, mode='a', encoding='utf-8')
failure_handler.setFormatter(logging.Formatter('%(message)s'))
failure_logger.addHandler(failure_handler)
# ============ Load choose_box.json ============
if not os.path.exists(json_path):
    failure_logger.warning(f"choose_box.json not found: {json_path}")
    raise SystemExit(1)
with open(json_path, 'r', encoding='utf-8') as f:
    try:
        choose_box_data = json.load(f)
    except json.JSONDecodeError as e:
        failure_logger.warning(f"JSON parse error: {e}")
        raise SystemExit(1)
# ============ CV Model Function ============
def CV(LSF, img, nu, mu, epsilon, step):
    Drc = (epsilon / math.pi) / (epsilon * epsilon + LSF * LSF)
    Hea = 0.5 * (1 + (2 / math.pi) * np.arctan(LSF / epsilon))
    Iy, Ix = np.gradient(LSF)
    s  = np.clip(np.sqrt(Ix * Ix + Iy * Iy), 1e-6, np.inf)
    Nx, Ny = Ix / s, Iy / s
    Mxx, Nxx = np.gradient(Nx)
    Nyy, Myy = np.gradient(Ny)
    curvature = Nxx + Nyy
    Length = nu * Drc * curvature
    Area   = mu * Drc
    s1, s2 = Hea * img, (1 - Hea) * img
    s3     = 1 - Hea
    C1, C2 = s1.sum() / Hea.sum(), s2.sum() / s3.sum()
    CVterm = Drc * (-1 * (img - C1) ** 2 + (img - C2) ** 2)
    LSF = LSF + step * (Length + Area + CVterm)
    return LSF
# ---- CV Hyperparameters ----
nu   = 0.01 * 255 * 255
mu   = 0.005 * 255 * 255
num_iterations = 30
epsilon = 10
step    = 1
# ============ Merge existing CV results ============
if os.path.exists(cv_bbox_json_path):
    with open(cv_bbox_json_path, 'r', encoding='utf-8') as f_cv:
        try:
            cv_bbox_data = json.load(f_cv)
            for cv_item in cv_bbox_data:
                for choose_item in choose_box_data:
                    if (choose_item["file_name"] == cv_item["file_name"] and
                            choose_item["bbox"] == cv_item["bbox"]):
                        if "new_bbox" in cv_item:
                            choose_item["new_bbox"] = cv_item["new_bbox"]
                        break
        except json.JSONDecodeError as e:
            failure_logger.warning(f"CV_bbox.json parse error: {e}")
            raise SystemExit(1)
# ============ again_picture / no_change log files ============
again_picture_path = os.path.join(current_dir, 'again_picture.txt')
again_picture_file = open(again_picture_path, 'a', encoding='utf-8')
no_change_path  = os.path.join(current_dir, 'no_change.txt')
no_change_file  = open(no_change_path, 'a', encoding='utf-8')
# ============ Whether process only files in no_change.txt ============
if PROCESS_ONLY_NO_CHANGE:
    with open(no_change_path, 'r', encoding='utf-8') as file:
        no_change_files = [line.strip() for line in file.readlines()]
else:
    no_change_files = []
# ============ Main Processing Loop ============
grouped_data = defaultdict(list)
for item in choose_box_data:
    grouped_data[item["file_name"]].append(item)
for file_name, items in grouped_data.items():
    if PROCESS_ONLY_NO_CHANGE and file_name not in no_change_files:
        continue
    # 1) Original image path
    original_image_path = os.path.join(image_folder, *file_name.split('/'))
    if not os.path.exists(original_image_path):
        failure_logger.warning(f"{file_name}: original image not found {original_image_path}")
        for item in items:
            item.setdefault("new_bbox", item["bbox"])
        continue
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        failure_logger.warning(f"{file_name}: failed to read image {original_image_path}")
        for item in items:
            item.setdefault("new_bbox", item["bbox"])
        continue
    # === Draw all bboxes for visualization ===
    bbox_image = original_image.copy()
    for idx, item in enumerate(items):
        if SKIP_EXISTING_NEW_BBOX and "new_bbox" in item:
            print(f"{file_name} - re{idx + 1}: new_bbox exists, skip")
            continue
        x0, y0, w0, h0 = item["bbox"]
        if w0 < 0: x0, w0 = x0 + w0, -w0
        if h0 < 0: y0, h0 = y0 + h0, -h0
        x0, y0 = max(x0, 0), max(y0, 0)
        # Red box: original bbox
        cv2.rectangle(bbox_image, (int(x0), int(y0)), (int(x0 + w0), int(y0 + h0)), (0, 0, 255), 2)
        # Blue box: existing new_bbox
        if "new_bbox" in item:
            nx, ny, nw, nh = item["new_bbox"]
            cv2.rectangle(bbox_image, (int(nx), int(ny)), (int(nx + nw), int(ny + nh)), (255, 0, 0), 2)
        cv2.putText(bbox_image, f"re{idx+1}", (int(x0), int(y0) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # === 【Modification A】: name bbox image by cropped_image basename ===
    cropped_base = os.path.splitext(os.path.basename(items[0]["cropped_image"]))[0]
    bbox_path    = os.path.join(os.path.dirname(original_image_path), f"{cropped_base}_bbox.png")
    cv2.imwrite(bbox_path, bbox_image)
    print(f"Saving bbox visualization -> {bbox_path}")
    # ========== Run CV for each bounding box ===============
    for idx, item in enumerate(items):
        if SKIP_EXISTING_NEW_BBOX and "new_bbox" in item:
            continue
        # ---- Coordinate normalization ----
        x, y, w, h = item["bbox"]
        if w < 0: x, w = x + w, -w
        if h < 0: y, h = y + h, -h
        x, y = max(x, 0), max(y, 0)
        w = min(w, original_image.shape[1] - x)
        h = min(h, original_image.shape[0] - y)
        if w <= 0 or h <= 0:
            failure_logger.warning(f"{file_name}-re{idx+1}: invalid bbox {item['bbox']}")
            item["new_bbox"] = [x, y, w, h]   # fallback
            no_change_file.write(f"{file_name} - re{idx+1}\n")
            again_picture_file.write(f"{file_name} - re{idx+1}\n")
            continue
        # ============ 【Modification A.1】: compute enlarged_bbox ============
        enlarge_w  = int(round(w * (1 + 2 * ENLARGE_RATIO)))
        enlarge_h  = int(round(h * (1 + 2 * ENLARGE_RATIO)))
        enlarge_x  = int(round(x - w * ENLARGE_RATIO))
        enlarge_y  = int(round(y - h * ENLARGE_RATIO))
        enlarge_x  = max(enlarge_x, 0)
        enlarge_y  = max(enlarge_y, 0)
        enlarge_w  = min(enlarge_w, original_image.shape[1] - enlarge_x)
        enlarge_h  = min(enlarge_h, original_image.shape[0] - enlarge_y)
        if enlarge_w <= 0 or enlarge_h <= 0:
            # Degenerate case, safely fallback
            enlarge_x, enlarge_y, enlarge_w, enlarge_h = x, y, w, h
        # --- Fallback bbox keeps original bbox ---
        fallback_bbox = [x, y, w, h]
        # ---- Crop grayscale image inside enlarged_bbox ----
        crop_gray = cv2.cvtColor(
            original_image[enlarge_y:enlarge_y + enlarge_h, enlarge_x:enlarge_x + enlarge_w],
            cv2.COLOR_BGR2GRAY
        )
        if crop_gray is None or crop_gray.size == 0:
            failure_logger.warning(f"{file_name}-re{idx+1}: crop failed enlarge_bbox={enlarge_x,enlarge_y,enlarge_w,enlarge_h}")
            item["new_bbox"] = fallback_bbox
            no_change_file.write(f"{file_name} - re{idx+1}\n")
            again_picture_file.write(f"{file_name} - re{idx+1}\n")
            continue
        # ---- Ellipse initialization for level set ----
        height, width = crop_gray.shape
        center_x, center_y = width // 2, height // 2
        a, b = max(width // 3, 1), max(height // 3, 1)
        IniLSF = np.ones((height, width), np.float32) * -1
        yy, xx = np.ogrid[:height, :width]
        mask_ellip = (((yy-center_y)**2)/b**2 + ((xx-center_x)**2)/a**2) <= 1
        IniLSF[mask_ellip] = 1
        LSF = IniLSF
        # ---- CV level‑set iteration ----
        for _ in range(num_iterations):
            LSF = CV(LSF, crop_gray, nu, mu, epsilon, step)
        # ---- Post‑processing ----
        LSF_smoothed = gaussian_filter(LSF, sigma=8)
        binary_LSF   = (LSF_smoothed >= 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_LSF, 8)
        min_area = 2000
        final_contour = np.zeros_like(binary_LSF)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                final_contour[labels == label] = 1
        kernel = np.ones((7, 7), np.uint8)
        final_contour = cv2.erode(final_contour, kernel, 3)
        final_contour = cv2.dilate(final_contour, kernel, 3)
        contours, _ = cv2.findContours(final_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            failure_logger.warning(f"{file_name}-re{idx+1}: no valid contour, fallback to original box")
            item["new_bbox"] = fallback_bbox
            no_change_file.write(f"{file_name} - re{idx+1}\n")
            again_picture_file.write(f"{file_name} - re{idx+1}\n")
            continue
        # ---- Pick largest contour ----
        largest_contour = max(contours, key=cv2.contourArea)
        # ---- Draw & save visualization ----
        result_image = original_image.copy()
        largest_mapped = largest_contour + np.array([enlarge_x, enlarge_y])   # 【Modification ②】
        cv2.drawContours(result_image, [largest_mapped], -1, (255, 0, 0), 2)
        # === 【Modification B】: name result image by cropped_image basename ===
        result_base  = os.path.splitext(os.path.basename(item["cropped_image"]))[0]
        result_path  = os.path.join(os.path.dirname(original_image_path),
                                    f"{result_base}_result.png")
        cv2.imwrite(result_path, result_image)
        print(f"Saving result image -> {result_path}")
        # ---- Compute new bbox ----
        x_new, y_new, w_new, h_new = cv2.boundingRect(largest_contour)
        # Area too small → fallback
        if w_new * h_new < (w * h) / 8:
            item["new_bbox"] = fallback_bbox
            no_change_file.write(f"{file_name} - re{idx+1}\n")
            print(f"{file_name}-re{idx+1}: new bbox area too small → keep original box")
        else:
            # 【Modification B.1】: add enlarged_bbox offset to get original‑image coordinates
            item["new_bbox"] = [enlarge_x + x_new, enlarge_y + y_new, w_new, h_new]
            print(f"{file_name}-re{idx+1}: update new_bbox = {item['new_bbox']}")
        again_picture_file.write(f"{file_name} - re{idx+1}\n")
    # ==== Save JSON output =========================================================
    try:
        with open(cv_bbox_json_path, 'w', encoding='utf-8') as f_out:
            json.dump(choose_box_data, f_out, ensure_ascii=False, indent=4)
    except Exception as e:
        failure_logger.warning(f"Failed to save CV_bbox json: {e}\n{traceback.format_exc()}")
# ============ Cleanup file handles ============
again_picture_file.close()
no_change_file.close()
print("\nProcessing finished! Check defeat_picture.log for failures.")
