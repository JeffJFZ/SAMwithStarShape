# -*- coding: utf-8 -*-
"""
  Author : Shoujun Huang, Junjie Liu, Shousheng Luo, Huafeng Xie, Jing Yuan, Dexing Kong, Jianfeng Zhang
  Usage  : python sourceCode.py
  Reference  : Our Paper in Biomedical Signal Processing and Control:
    From automatic detection to segmentation: A label-efficient SAM pipeline with feature-guided star-shape priors for breast ultrasound 
"""

from __future__ import annotations
import os
import re
import cv2
import numpy as np
import numpy.fft as nf
from typing import Tuple, List
# --------------------------------------------------------------------
#                      Global Configuration & Hyper‑parameters
# --------------------------------------------------------------------
EPS              = 1e-6
MAX_ITERS        = 50
SHOW_STEP        = 50
TEMP             = 0.2
PROB_TH                = 0.45
WHITE_RATIO_THRESHOLD  = 0.4487
STAR_WEIGHT   = 1.0
REGION_WEIGHT = 1.0
BASE_BETA        = 1
BASE_LBDA        = None
ALPHA_FEAT_BASE  = 1.0
W_IN, W_OUT = 1, 1
ELLIPSE_PAD_RATIO = 0.15
# ---------- Morphology kernels ----------
MORPH_KERNEL       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
GAP_KERNEL_SIZE    = 30
GAP_CLOSE_KERNEL   = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    ((GAP_KERNEL_SIZE // 2) * 2 + 1, ) * 2)
PATIENCE           = 5
IOU_EPS            = 1e-3
TORT_THRESHOLD     = 1.2
# -------------------- Dense‑CRF + AAF --------------------
def _try_import_densecrf():
    try:
        import pydensecrf.densecrf as dcrf
        import pydensecrf.utils as dcrf_utils
        return dcrf, dcrf_utils
    except ModuleNotFoundError:
        return None, None
_DCRF, _DCRF_UTILS = _try_import_densecrf()
def apply_dense_crf(img_bgr: np.ndarray,
                    prob: np.ndarray,
                    n_iters: int = 5) -> np.ndarray:
    if _DCRF is None:
        return prob
    h, w = prob.shape
    dcrf = _DCRF.DenseCRF2D(w, h, 2)
    unary = np.stack([1.0 - prob, prob], axis=0)
    unary = -np.log(np.clip(unary, 1e-6, 1.0)).reshape(2, -1)
    dcrf.setUnaryEnergy(unary.astype(np.float32))
    _DCRF_UTILS.add_pairwise_gaussian(dcrf, sxy=3, compat=3)
    _DCRF_UTILS.add_pairwise_bilateral(dcrf, img_bgr, sxy=60, srgb=5, compat=5)
    Q = dcrf.inference(n_iters)
    return np.array(Q[1]).reshape(h, w).astype(np.float32)
def apply_aaf(mask: np.ndarray) -> np.ndarray:
    smooth = cv2.bilateralFilter(mask, d=9, sigmaColor=75, sigmaSpace=75)
    return cv2.medianBlur(smooth, 5)
# --------------------------------------------------------------------
# utility：Select Top‑4 informative channels by Δμ metric
# --------------------------------------------------------------------
def get_topk_channel_weights(F: np.ndarray,
                             roi_mask: np.ndarray,
                             k: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    in_feat  = F[roi_mask]
    out_feat = F[~roi_mask]
    if in_feat.size == 0 or out_feat.size == 0:
        picks = np.arange(k, dtype=np.int32)
    else:
        scores = np.abs(in_feat.mean(0) - out_feat.mean(0))
        picks  = np.argpartition(-scores, k)[:k]
    w = np.zeros(F.shape[2], np.float32)
    w[picks] = 1.0 / k
    return w.reshape(1, 1, -1), picks
# --------------------------------------------------------------------
# util：Keep largest connected component
# --------------------------------------------------------------------
def keep_largest_component(bin_mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (bin_mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return bin_mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)
# --------------------------------------------------------------------
# NEW util：Fill all inner holes
# --------------------------------------------------------------------
def fill_all_holes(bin_mask: np.ndarray) -> np.ndarray:
    h, w = bin_mask.shape
    fg = (bin_mask > 0).astype(np.uint8)
    bg = 1 - fg
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)
    filled = fg.copy()
    for i in range(1, n):
        x, y, w_, h_, _ = stats[i]
        border = (x == 0 or y == 0 or x + w_ == w or y + h_ == h)
        if not border:
            filled[labels == i] = 1
    return (filled * 255).astype(np.uint8)
# --------------------------------------------------------------------
# NEW util：Selective close operation (fill gaps while limiting outward expansion)
# --------------------------------------------------------------------
def selective_close(mask: np.ndarray,
                    gap_kernel: np.ndarray,
                    safety_pad: int = 5) -> np.ndarray:
    """
    1. Execute close operation with gap_kernel to fill wide gaps;
    2. Dilate original mask by safety_pad to obtain safety_zone;
    3. Only preserve closed ∩ safety_zone to prevent out‑contour spurious tentacles.
    """
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, gap_kernel)
    safety_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        ((safety_pad * 2 + 1, ) * 2))
    safety_zone = cv2.dilate(mask, safety_kernel)
    return cv2.bitwise_and(closed, safety_zone)
# --------------------------------------------------------------------
#                    Tortuosity measurement
# --------------------------------------------------------------------
def measure_tortuosity(mask: np.ndarray) -> float:
    thresh = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 1.0
    cnt       = max(cnts, key=cv2.contourArea)
    peri      = cv2.arcLength(cnt, True)
    hull_peri = cv2.arcLength(cv2.convexHull(cnt), True)
    return max(peri / hull_peri, 1.0) if hull_peri else 1.0
# ---------------- helper: white‑area ratio → λ ---------------------
def calc_lbda(ratio: float) -> float:
    if ratio < 0.05:   return 0.03
    if ratio < 0.20:   return 0.05
    if ratio < 0.40:   return 0.07
    return 0.09
# --------------------------------------------------------------------
#              Utility functions: vector field & multi‑scale features
# --------------------------------------------------------------------
def compute_vector_field(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    edges = cv2.Canny(raw, 100, 200)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mc = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(mc, True)
    pts_num, interval, curd, pts = 30, peri / 30, 0.0, []
    for i in range(len(mc) - 1):
        p0, p1 = mc[i][0], mc[i+1][0]
        seg = np.linalg.norm(p1 - p0)
        curd += seg
        while curd >= interval and len(pts) < pts_num:
            pts.append(p0); curd -= interval
        if len(pts) >= pts_num:
            break
    P = np.array(pts)
    H, W = raw.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    idx = np.argmin((X[..., None] - P[:, 0])**2 + (Y[..., None] - P[:, 1])**2, axis=2)
    vx = P[idx, 0] - X
    vy = P[idx, 1] - Y
    norm = np.sqrt(vx**2 + vy**2) + 1e-5
    return -vx / norm, -vy / norm
def compute_fft_denominator(H: int, W: int) -> np.ndarray:
    U, V = np.meshgrid(np.arange(W), np.arange(H))
    return 2 * (2 - np.cos(2*np.pi*U/W) - np.cos(2*np.pi*V/H)) + 0.05
def multi_scale_features(F: np.ndarray, scales: List[float] = [0.5, 1.0, 2.0]) -> np.ndarray:
    H, W, C = F.shape
    out = []
    for s in scales:
        if s != 1.0:
            new_H, new_W = int(H * s), int(W * s)
            resized = np.stack([cv2.resize(F[:, :, c], (new_W, new_H), interpolation=cv2.INTER_LINEAR)
                                for c in range(C)], axis=2)
            resized = cv2.resize(resized, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            resized = F
        out.append(resized)
    return np.mean(np.stack(out, axis=0), axis=0)
def attention_mechanism(F: np.ndarray, u: np.ndarray) -> np.ndarray:
    att = u[:, :, 0]
    att = (att - att.min()) / (att.max() - att.min() + 1e-6)
    return F * att[..., None]
# --------------------------------------------------------------------
#                       Core processing function
# --------------------------------------------------------------------
def process_image(imgn: str,
                  mask_folder: str,
                  output_folder: str,
                  picture_folder: str,
                  iterations: int | None = None):
    global STAR_WEIGHT, REGION_WEIGHT
    mask_path = os.path.join(mask_folder, f"{imgn}.png")
    raw_mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if raw_mask is None:
        raise FileNotFoundError(mask_path)
    # ================================================================
    # Preprocessing: selective close (fill ~15px wide gaps, max outward expansion ≤5px)
    # ================================================================
    raw_mask_pre = selective_close(raw_mask, GAP_CLOSE_KERNEL, safety_pad=5)
    white_ratio = (raw_mask_pre > 0).mean()
    if white_ratio > WHITE_RATIO_THRESHOLD:
        STAR_WEIGHT, REGION_WEIGHT = 1.0, 1.0
        iterations = iterations or 100
    else:
        tort = measure_tortuosity(raw_mask_pre)
        STAR_WEIGHT, REGION_WEIGHT = (1.0, 1.0) if tort < TORT_THRESHOLD else (0.5, 1.5)
        iterations = iterations or MAX_ITERS
    print(f"[INFO] {imgn}: STAR_W={STAR_WEIGHT}, REGION_W={REGION_WEIGHT}, iters={iterations}")
    H, W = raw_mask_pre.shape
    lam = calc_lbda(white_ratio)
    fx, fy = compute_vector_field(raw_mask_pre)
    dom = compute_fft_denominator(H, W)[..., None]
    u = np.zeros((H, W, 2), np.float32)
    u[:, :, 0] = np.where(raw_mask_pre > 200, 0.95,
                   np.where(raw_mask_pre > 0, 0.7, 0.05))
    u[:, :, 1] = 1 - u[:, :, 0]
    # Fine‑kernel close to smooth boundary
    mask_closed = cv2.morphologyEx(raw_mask_pre, cv2.MORPH_CLOSE, MORPH_KERNEL)
    mask_bin = (mask_closed > 0).astype(np.float32)
    ys, xs = np.where(mask_bin > 0.5)
    if len(xs) < 5:
        raise ValueError('mask too small')
    (cx, cy), (MA, ma), ang = cv2.fitEllipse(np.column_stack((xs, ys)).astype(np.float32))
    pad = int(ELLIPSE_PAD_RATIO * min(MA, ma))
    ellipse_mask = np.zeros_like(mask_bin, np.uint8)
    cv2.ellipse(ellipse_mask, (int(cx), int(cy)),
                (int(MA/2 + pad), int(ma/2 + pad)),
                ang, 0, 360, 255, -1)
    roi_mask = ellipse_mask.astype(bool)
    root_dir   = os.path.dirname(mask_folder)
    orig_name  = mask_to_orig(imgn)
    feat_path  = os.path.join(root_dir, 'feature_map', f"{orig_name}_feats.npy")
    feats      = np.load(feat_path)
    if feats.ndim == 4 and feats.shape[0] == 1:
        feats = feats[0]
    C, _, _ = feats.shape
    F = np.zeros((H, W, C), np.float32)
    for c in range(C):
        F[..., c] = cv2.resize(feats[c], (W, H), interpolation=cv2.INTER_LINEAR)
    F = multi_scale_features(F)
    channel_weights, picks = get_topk_channel_weights(F, roi_mask, k=8)
    print(f"[INFO] {imgn}: selected channels → {sorted(picks.tolist())}")
    ux, uy = np.zeros_like(u), np.zeros_like(u)
    def compute_gradient():
        uy[:-1]    = u[1:] - u[:-1]
        ux[:, :-1] = u[:, 1:] - u[:, :-1]
        uy[-1]     = u[0]  - u[-1]
        ux[:, -1]  = u[:, 0] - u[:, -1]
    compute_gradient()
    prev_seg, stale = None, 0
    for k in range(1, iterations + 1):
        # dual update
        inner1 = ux[:, :, 0] * fx + uy[:, :, 0] * fy
        inner2 = -ux[:, :, 0] * fy + uy[:, :, 0] * fx
        grad = np.sqrt(ux**2 + uy**2)
        shrink = np.maximum(grad - lam * STAR_WEIGHT, 0)
        m = grad > 0
        shrink[m] /= grad[m]
        ux *= shrink
        uy *= shrink
        dx = np.maximum(np.abs(inner2) - lam * STAR_WEIGHT, 0) * np.sign(inner2)
        ux[:, :, 0][inner1 < 0] = -dx[inner1 < 0] * fy[inner1 < 0]
        uy[:, :, 0][inner1 < 0] =  dx[inner1 < 0] * fx[inner1 < 0]
        # primal update (star‑convex constraint)
        divg = np.zeros_like(u)
        divg[:-1]     += uy[:-1];   divg[1:]      -= uy[:-1]
        divg[:, :-1]  += ux[:, :-1]; divg[:, 1:]   -= ux[:, :-1]
        u_star = nf.ifft2(nf.fft2(0.05 * u - divg, axes=(0, 1)) / dom,
                          axes=(0, 1)).real
        # region energy term
        inside  = (u[:, :, 0] > 0.5) & roi_mask
        outside = (~inside) & roi_mask
        mu_in  = F[inside].mean(0)  if inside.any()  else np.zeros(C)
        mu_out = F[outside].mean(0) if outside.any() else np.zeros(C)
        mu_in, mu_out = mu_in.reshape(1,1,C), mu_out.reshape(1,1,C)
        d_in  = W_IN  * np.sum(channel_weights * (F - mu_in )**2, axis=2)
        d_out = W_OUT * np.sum(channel_weights * (F - mu_out)**2, axis=2)
        data = np.stack([d_in*(0.1+0.9*roi_mask),
                         d_out*(0.1+0.9*roi_mask)], axis=2) * ALPHA_FEAT_BASE * REGION_WEIGHT
        exp_u = np.exp(-(0.5 - u + BASE_BETA * data) / TEMP)
        u_region = exp_u / (exp_u.sum(2, keepdims=True) + EPS)
        # attention mechanism
        F = attention_mechanism(F, u)
        # adaptive fusion
        confidence = np.mean(u[:, :, 0] * (1 - u[:, :, 0]))
        w_star = 0.5 + 0.3 * (1 - confidence)
        w_region = 1 - w_star
        u = w_star * u_star + w_region * u_region
        compute_gradient()
        seg_curr = (u[:, :, 0] > PROB_TH).astype(np.uint8)
        if prev_seg is not None:
            inter = (seg_curr & prev_seg).sum()
            union = (seg_curr | prev_seg).sum()
            stale = stale + 1 if 1.0 - inter / (union + EPS) < IOU_EPS else 0
        prev_seg = seg_curr
        if k % SHOW_STEP == 0 or k == iterations:
            print(f"  ▸ iter {k}/{iterations}; stale={stale}")
        if stale >= PATIENCE:
            print(f"[EARLY STOP] at iter {k}")
            break
    prob_map = u[:, :, 0].astype(np.float32)
    pic_path = os.path.join(picture_folder, f"{orig_name}.png")
    img_bgr  = cv2.imread(pic_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # ------------------- Final post‑processing -------------------
    prob_ref = apply_dense_crf(img_bgr, prob_map, n_iters=5)
    seg_final = (prob_ref > PROB_TH).astype(np.uint8) * 255
    seg_final = apply_aaf(seg_final)
    # Critical: wide‑kernel close to seal open contour gaps
    seg_final = cv2.morphologyEx(seg_final, cv2.MORPH_CLOSE, GAP_CLOSE_KERNEL)
    # Retain only largest connected block after closing
    seg_final = keep_largest_component(seg_final)
    # Fill all inner holes
    seg_final = fill_all_holes(seg_final)
    # Mild boundary smoothing
    seg_final = cv2.morphologyEx(seg_final, cv2.MORPH_CLOSE, MORPH_KERNEL)
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"{imgn}.png")
    cv2.imwrite(out_path, seg_final)
    print('[OK] saved', out_path)
# --------------------------------------------------------------------
#                  Batch helper & entry point
# --------------------------------------------------------------------
_mask_pat = re.compile(r"^(.*?\))")
def mask_to_orig(mask_name: str) -> str:
    m = _mask_pat.match(mask_name)
    return m.group(1) if m else mask_name.split('_')[0]
def run_folder(root: str, mask_sub: str, out_sub: str, picture_root: str):
    mask_folder = os.path.join(root, mask_sub)
    out_folder  = os.path.join(root, out_sub)
    for fn in os.listdir(mask_folder):
        if not fn.lower().endswith(('.png', '.jpg')):
            continue
        name = os.path.splitext(fn)[0]
        print('\n=====', fn, '=====')
        try:
            process_image(name, mask_folder, out_folder, picture_root)
        except KeyboardInterrupt:
            print('\n[INTERRUPT] user aborted batch.')
            return
        except Exception as e:
            print('[ERROR]', name, ':', e)
def main():
    root = r"PATH\BUSI_compare"
    picture_root = os.path.join(root, 'BUSI_picture')
    run_folder(root, 'new_bbox', 'new_bbox_star', picture_root)
if __name__ == '__main__':
    main()
