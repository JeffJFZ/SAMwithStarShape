# -*- coding: utf-8 -*-
"""
  Author : Shoujun Huang, Junjie Liu, Shousheng Luo, Huafeng Xie, Jing Yuan, Dexing Kong, Jianfeng Zhang
  Usage  : python sourceCode.py
  Reference  : Our Paper in Biomedical Signal Processing and Control:
    From automatic detection to segmentation: A label-efficient SAM pipeline with feature-guided star-shape priors for breast ultrasound 
"""

import os
import glob
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
# ------------------------
# Configuration
# ------------------------
sam_checkpoint = r"models/sam_vit_h_4b8939.pth"
model_type     = "vit_h"
device         = "cuda"
image_dir   = r"PATH\BUSI_picture"
feature_dir = r"PATH\feature_map"
os.makedirs(feature_dir, exist_ok=True)
# ------------------------
# Load SAM & Predictor
# ------------------------
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
# ------------------------
# Iterate images, extract and save features
# ------------------------
max_image_size = 1024
img_paths = glob.glob(os.path.join(image_dir, "*.png")) + \
            glob.glob(os.path.join(image_dir, "*.jpg"))
for img_path in img_paths:
    print(f"[INFO] Processing：{img_path}")
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"[WARN] Read image failed, skip：{img_path}")
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(max_image_size / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        rgb = cv2.resize(rgb, (new_w, new_h))
        print(f"[INFO] Resize image to：{new_w}x{new_h}")
    try:
        torch.cuda.empty_cache()
        predictor.set_image(rgb)
        feats: torch.Tensor = predictor.get_image_embedding()
        feats_np = feats.cpu().numpy()
        base = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(feature_dir, f"{base}_feats.npy")
        np.save(save_path, feats_np)
        print(f"[OK] Feature saved：{save_path}")
    except Exception as e:
        print(f"[ERROR] Processing failed：{img_path}")
        print(f"        Error info：{e}")
        continue
print("Feature extraction for all images finished.")
