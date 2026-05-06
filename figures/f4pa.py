"""
Figure 4, Panel A — Cellpose Segmentation Evaluation
Architecture:
  Phase 1: Run cellpose ONCE → cache all masks to H5
  Phase 2: For each matching condition, compute metrics from cached masks
  Phase 3: Select best threshold (highest PQ for Ours), plot two PDFs

Outputs:
  - Figure_4_Panel_a.pdf: Left=visual, Right=PQ violin only
  - Sup_Fig_4.pdf: F1 and Dice violin plots
"""

import os
import sys
import argparse
import copy
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
import h5py
import cv2
import torch
import torch.nn.functional as F
from cellpose import models
from tqdm import tqdm
from scipy import stats
import nature_style

# ──────────────────────────────────────────────────────────────
# 0. Configuration
# ──────────────────────────────────────────────────────────────
logger = logging.getLogger('cellpose')
logger.setLevel(logging.ERROR)

mm = nature_style.apply_nature_style()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Matching conditions to evaluate
MATCH_CONDITIONS = [
    ('iou',      0.20),
    ('iou',      0.25),
    ('iou',      0.30),
    ('iou',      0.50),
    ('centroid',  3.0),
    ('centroid',  5.0),
    ('centroid', 10.0),
]

# Cellpose & evaluation parameters
DIAMETER = 30
CELLPOSE_BATCH_SIZE = 64          # Maximize GPU util (80GB VRAM)
CENTER_START, CENTER_END = 5, 15  # Evaluate central 10 slices
SHIFT_RANGE = 2                   # Z-shift matching range
NUM_THRES = 10                    # Min avg cell count to keep sample

# 3DSR HDF5 with `hr`, `lr`, and per-method prediction datasets
# (e.g. `microdiffuse3d`, `RCAN_output`, `RLN_output`, `SwinIR_output`).
H5_PATH = '<YOUR_DATA_PATH>'

# Cache paths
MASK_CACHE = 'f4pa_new_masks.h5'      # Cellpose masks (run once)
METRICS_CACHE_DIR = 'f4pa_new_caches'  # Per-condition CSV caches

# Visualization parameters
VIS_SLICE_IDX = 10

# Plot selection
PLOT_CONDITION = 'iou_0.3'  # Fixed: use IoU > 0.3 for plotting


# ──────────────────────────────────────────────────────────────
# 1. Helper functions
# ──────────────────────────────────────────────────────────────

def interpolate_lr_volume(lr_numpy, target_depth=20):
    """Trilinear interpolation of LR volume to match HR depth."""
    lr_tensor = torch.from_numpy(lr_numpy).float().to(device)
    if lr_tensor.ndim == 3:
        lr_tensor = lr_tensor.unsqueeze(0).unsqueeze(0)
    elif lr_tensor.ndim == 4:
        lr_tensor = lr_tensor.permute(1, 0, 2, 3).unsqueeze(0)
    H, W = lr_numpy.shape[-2:]
    out = F.interpolate(lr_tensor, size=(target_depth, H, W),
                        mode='trilinear', align_corners=False)
    return out.squeeze().cpu().numpy()


def prep_for_cellpose(imgs):
    """Normalize images to uint8 for cellpose."""
    return [cv2.normalize(i, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            for i in imgs]


def get_centroids(mask):
    """Return dict: {label_id: (row_centroid, col_centroid)}."""
    ids = np.unique(mask)
    ids = ids[ids != 0]
    centroids = {}
    for lid in ids:
        coords = np.argwhere(mask == lid)
        centroids[lid] = coords.mean(axis=0)
    return centroids


def compute_match_stats(pred_mask, gt_mask, mode='iou', threshold=0.5):
    """
    Compute TP/FP/FN and per-match IoU using either IoU or centroid matching.
    Returns: tp, fp, fn, matched_ious
    """
    pred_ids = np.unique(pred_mask)[1:]
    gt_ids = np.unique(gt_mask)[1:]

    tp = 0
    matched_gt = set()
    matched_ious = []

    if mode == 'iou':
        for pid in pred_ids:
            mask_p = (pred_mask == pid)
            intersect = gt_mask[mask_p]
            if intersect.size == 0:
                continue
            ids, counts = np.unique(intersect, return_counts=True)
            valid = ids != 0
            if not valid.any():
                continue
            best_gt = ids[valid][np.argmax(counts[valid])]
            mask_g = (gt_mask == best_gt)
            iou = np.sum(mask_p & mask_g) / np.sum(mask_p | mask_g)
            if iou > threshold and best_gt not in matched_gt:
                tp += 1
                matched_gt.add(best_gt)
                matched_ious.append(iou)

    elif mode == 'centroid':
        pred_centroids = get_centroids(pred_mask)
        gt_centroids = get_centroids(gt_mask)
        pairs = []
        for pid, pc in pred_centroids.items():
            for gid, gc in gt_centroids.items():
                dist = np.linalg.norm(pc - gc)
                pairs.append((dist, pid, gid))
        pairs.sort(key=lambda x: x[0])
        matched_pred = set()
        for dist, pid, gid in pairs:
            if pid in matched_pred or gid in matched_gt:
                continue
            if dist < threshold:
                tp += 1
                matched_pred.add(pid)
                matched_gt.add(gid)
                mask_p = (pred_mask == pid)
                mask_g = (gt_mask == gid)
                iou = np.sum(mask_p & mask_g) / (np.sum(mask_p | mask_g) + 1e-8)
                matched_ious.append(iou)

    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    return tp, fp, fn, matched_ious


# ──────────────────────────────────────────────────────────────
# 2. Phase 1: Run cellpose ONCE and cache masks
# ──────────────────────────────────────────────────────────────

# Rename methods for display
METHOD_DISPLAY = {
    'LR (Interp)': 'Interpolation',
    'RCAN_output': '3DRCAN',
    'RLN_output': 'RLN',
    'SwinIR_output': 'SwinIR',
    'microdiffuse3d': 'Ours',
}

def rename_method(name):
    for key, val in METHOD_DISPLAY.items():
        if key in name:
            return val
    return name


def run_cellpose_and_cache(h5_path, mask_cache_path, diameter=30,
                           vis_sample_idx=0, limit_samples=None):
    """
    Run cellpose on all methods ONCE. Save masks to H5.
    Returns: method_names, valid_indices, vis_data_dict, vis_sample_idx
    """
    model = models.CellposeModel(gpu=True, pretrained_model='./cpsam')


    with h5py.File(h5_path, 'r') as f:
        ds_hr = f['hr']
        ds_lr = f['lr']
        output_keys = [k for k in f.keys() if "output" in k or "sr" in k]
        num_samples = ds_hr.shape[0]
        if limit_samples:
            num_samples = min(num_samples, limit_samples)

        # Determine image shape from first sample
        sample_shape = ds_hr[0].squeeze()[0].shape  # (H, W)
        H, W = sample_shape

        # All method names (including GT and LR)
        all_method_keys = ['Ground Truth', 'LR (Interp)'] + list(output_keys)

        # Open mask cache for writing
        with h5py.File(mask_cache_path, 'w') as mf:
            # Store raw images for visualization
            mf.attrs['num_samples'] = num_samples
            mf.attrs['num_slices'] = 20

            # Pre-allocate datasets for masks
            for mk in all_method_keys:
                safe_name = mk.replace(' ', '_').replace('(', '').replace(')', '')
                mf.create_dataset(
                    f'masks/{safe_name}',
                    shape=(num_samples, 20, H, W),
                    dtype=np.int32,
                    chunks=(1, 20, H, W),
                    compression='gzip')

            # Track which samples are valid (pass sparse filter)
            valid_flags = np.zeros(num_samples, dtype=bool)
            # Store raw images for vis sample
            vis_data_dict = {}
            vis_found = False

            for idx in tqdm(range(num_samples), desc="Cellpose"):
                hr_vol = ds_hr[idx].squeeze()
                lr_raw = ds_lr[idx].squeeze()
                lr_interp = interpolate_lr_volume(lr_raw, target_depth=20)

                batch_data = {
                    'Ground Truth': [hr_vol[z] for z in range(20)],
                    'LR (Interp)': [lr_interp[z] for z in range(20)]
                }
                for k in output_keys:
                    data = f[k][idx].squeeze()
                    if len(data.shape) == 3 and data.shape[0] == 20:
                        batch_data[k] = [data[z] for z in range(20)]

                # --- Batch cellpose: group by parameter set ---
                default_imgs = []  # (method_name, slice_imgs)
                tuned_imgs = [] # if need specific params for cellpose (not activated)

                default_imgs.append(('Ground Truth', batch_data['Ground Truth']))
                for method_name, imgs_list in batch_data.items():
                    if method_name == 'Ground Truth':
                        continue
                    # GT, LR, and SiT use default cellpose params
                    # if "sit" in method_name or "LR" in method_name:
                    default_imgs.append((method_name, imgs_list))
                    # else:
                    #     tuned_imgs.append((method_name, imgs_list))

                # Flatten into one big list per group
                default_flat = []
                default_index = []  # (method_name, start_idx, count)
                for mname, slices in default_imgs:
                    start = len(default_flat)
                    prepped = prep_for_cellpose(slices)
                    default_flat.extend(prepped)
                    default_index.append((mname, start, len(prepped)))

                tuned_flat = []
                tuned_index = []
                for mname, slices in tuned_imgs:
                    start = len(tuned_flat)
                    prepped = prep_for_cellpose(slices)
                    tuned_flat.extend(prepped)
                    tuned_index.append((mname, start, len(prepped)))

                # Run cellpose in two batched calls
                default_masks_all, _, _ = model.eval(
                    default_flat, diameter=diameter, channels=None,
                    batch_size=CELLPOSE_BATCH_SIZE)

                # Extract GT masks first for sparse filtering
                gt_start = default_index[0][1]
                gt_count = default_index[0][2]
                gt_masks_list = default_masks_all[gt_start:gt_start + gt_count]

                # Sparse filtering based on GT cell count
                central_counts = [len(np.unique(gt_masks_list[z])) - 1
                                  for z in range(CENTER_START, CENTER_END)]
                if np.mean(central_counts) < NUM_THRES:
                    continue

                valid_flags[idx] = True

                # Save GT masks
                for z in range(20):
                    mf['masks/Ground_Truth'][idx, z] = gt_masks_list[z]

                # Save other default-param method masks
                for mname, start, count in default_index[1:]:
                    safe_name = mname.replace(' ', '_').replace('(', '').replace(')', '')
                    masks = default_masks_all[start:start + count]
                    for z in range(count):
                        mf[f'masks/{safe_name}'][idx, z] = masks[z]

                # Run tuned-params group
                if tuned_flat:
                    tuned_masks_all, _, _ = model.eval(
                        tuned_flat, diameter=diameter, channels=None,
                        cellprob_threshold=-2.0, flow_threshold=2.0,
                        batch_size=CELLPOSE_BATCH_SIZE)

                    for mname, start, count in tuned_index:
                        safe_name = mname.replace(' ', '_').replace('(', '').replace(')', '')
                        masks = tuned_masks_all[start:start + count]
                        for z in range(count):
                            mf[f'masks/{safe_name}'][idx, z] = masks[z]

                # Save vis data
                if idx == vis_sample_idx or (not vis_found and vis_sample_idx is None):
                    vis_data_dict = copy.deepcopy(batch_data)
                    vis_sample_idx = idx
                    vis_found = True

            # Store valid flags and method keys
            mf.create_dataset('valid_flags', data=valid_flags)
            # Store method key names as attributes
            for i, mk in enumerate(all_method_keys):
                mf.attrs[f'method_{i}'] = mk
            mf.attrs['num_methods'] = len(all_method_keys)
            mf.attrs['vis_sample_idx'] = vis_sample_idx


    return vis_data_dict, vis_sample_idx


# ──────────────────────────────────────────────────────────────
# 3. Phase 2: Compute metrics from cached masks
# ──────────────────────────────────────────────────────────────

def compute_metrics_from_cache(mask_cache_path, match_mode, match_threshold):
    """
    Load cached masks, apply matching, compute metrics.
    Returns: DataFrame with per-slice metrics
    """
    all_metrics = []

    with h5py.File(mask_cache_path, 'r') as mf:
        num_samples = mf.attrs['num_samples']
        valid_flags = mf['valid_flags'][:]
        num_methods = mf.attrs['num_methods']
        method_keys = [mf.attrs[f'method_{i}'] for i in range(num_methods)]

        gt_key = 'Ground_Truth'

        op = '>' if match_mode == 'iou' else '<'
        for idx in tqdm(range(num_samples),
                        desc=f"Metrics [{match_mode}{op}{match_threshold}]"):
            if not valid_flags[idx]:
                continue

            gt_masks = mf[f'masks/{gt_key}'][idx]  # (20, H, W)

            # GT self-reference
            for z in range(CENTER_START, CENTER_END):
                count = len(np.unique(gt_masks[z])) - 1
                all_metrics.append({
                    'Sample_Idx': idx, 'Slice_Idx': z,
                    'Method': 'Ground Truth',
                    'TP': count, 'FP': 0, 'FN': 0,
                    'Precision': 1.0, 'Recall': 1.0,
                    'F1': 1.0, 'Dice': 1.0, 'PQ': 1.0,
                    'Best_Shift': 0
                })

            # Each non-GT method
            for mk in method_keys:
                if mk == 'Ground Truth':
                    continue
                safe_name = mk.replace(' ', '_').replace('(', '').replace(')', '')
                pred_masks = mf[f'masks/{safe_name}'][idx]  # (20, H, W)

                # Global z-shift matching
                best_shift_f1 = -1.0
                best_shift_results = []

                for shift in range(-SHIFT_RANGE, SHIFT_RANGE + 1):
                    cur_metrics = []
                    for z_gt in range(CENTER_START, CENTER_END):
                        z_pred = z_gt + shift
                        if 0 <= z_pred < 20:
                            pred_slice = pred_masks[z_pred]
                            gt_slice = gt_masks[z_gt]

                            tp, fp, fn, matched_ious = compute_match_stats(
                                pred_slice, gt_slice,
                                mode=match_mode, threshold=match_threshold)
                            p = tp / (tp + fp + 1e-6)
                            r = tp / (tp + fn + 1e-6)
                            f1 = 2 * p * r / (p + r + 1e-6)

                            # Pixel-level Dice: 2|P∩G|/(|P|+|G|)
                            pred_fg = (pred_slice > 0).astype(np.float32)
                            gt_fg = (gt_slice > 0).astype(np.float32)
                            intersection = np.sum(pred_fg * gt_fg)
                            dice = 2 * intersection / (np.sum(pred_fg) + np.sum(gt_fg) + 1e-6)

                            dq = f1
                            sq = np.mean(matched_ious) if matched_ious else 0.0
                            pq = dq * sq

                            cur_metrics.append({
                                'Sample_Idx': idx, 'Slice_Idx': z_gt,
                                'Method': mk,
                                'TP': tp, 'FP': fp, 'FN': fn,
                                'Precision': p, 'Recall': r,
                                'F1': f1, 'Dice': dice, 'PQ': pq,
                                'Best_Shift': shift
                            })

                    if cur_metrics:
                        avg_f1 = np.mean([m['F1'] for m in cur_metrics])
                        if avg_f1 > best_shift_f1:
                            best_shift_f1 = avg_f1
                            best_shift_results = cur_metrics

                if best_shift_results:
                    all_metrics.extend(best_shift_results)

    return pd.DataFrame(all_metrics)


# ──────────────────────────────────────────────────────────────
# 4. Semantic mask for visualization
# ──────────────────────────────────────────────────────────────

COLOR_TP = np.array([0.0, 0.6, 0.53])    # Teal #009988
COLOR_FP = np.array([0.93, 0.47, 0.20])  # Coral #EE7733
COLOR_FN = np.array([0.67, 0.20, 0.47])  # Purple #AA3377
COLOR_GT = np.array([0.0, 0.47, 0.73])   # Steel blue #0077BB


def compute_semantic_mask(pred_mask, gt_mask, match_mode='iou',
                          match_threshold=0.5):
    """Generate TP/FP/FN RGB overlay."""
    h, w = pred_mask.shape
    rgb_map = np.zeros((h, w, 3), dtype=np.float32)

    pred_ids = np.unique(pred_mask)[1:]
    gt_ids = np.unique(gt_mask)[1:]
    tp_pred_ids = set()
    matched_gt_ids = set()

    if match_mode == 'iou':
        for pid in pred_ids:
            mask_p = (pred_mask == pid)
            intersect = gt_mask[mask_p]
            if intersect.size == 0:
                continue
            ids, counts = np.unique(intersect, return_counts=True)
            valid = ids != 0
            if not valid.any():
                continue
            best_gt = ids[valid][np.argmax(counts[valid])]
            mask_g = (gt_mask == best_gt)
            iou = np.sum(mask_p & mask_g) / np.sum(mask_p | mask_g)
            if iou > match_threshold and best_gt not in matched_gt_ids:
                tp_pred_ids.add(pid)
                matched_gt_ids.add(best_gt)
    elif match_mode == 'centroid':
        pred_centroids = get_centroids(pred_mask)
        gt_centroids = get_centroids(gt_mask)
        pairs = []
        for pid, pc in pred_centroids.items():
            for gid, gc in gt_centroids.items():
                pairs.append((np.linalg.norm(pc - gc), pid, gid))
        pairs.sort(key=lambda x: x[0])
        matched_pred = set()
        for dist, pid, gid in pairs:
            if pid in matched_pred or gid in matched_gt_ids:
                continue
            if dist < match_threshold:
                tp_pred_ids.add(pid)
                matched_pred.add(pid)
                matched_gt_ids.add(gid)

    fp_ids = set(pred_ids) - tp_pred_ids
    fn_ids = set(gt_ids) - matched_gt_ids

    for gid in fn_ids:
        rgb_map[gt_mask == gid] = COLOR_FN
    for pid in tp_pred_ids:
        rgb_map[pred_mask == pid] = COLOR_TP
    for pid in fp_ids:
        rgb_map[pred_mask == pid] = COLOR_FP

    return rgb_map, len(tp_pred_ids), len(fp_ids), len(fn_ids)


# ──────────────────────────────────────────────────────────────
# 5. Plotting helpers
# ──────────────────────────────────────────────────────────────

def add_significance(ax, data, metric='PQ', method_order=None,
                     ours_name='Ours'):
    """
    Add Wilcoxon significance bar: Ours vs best baseline.
    Best baseline is determined by PQ (primary metric) for consistency,
    then significance is tested on the specified metric.
    """
    if method_order is None:
        method_order = ['Interpolation', '3DRCAN', 'Ours']

    # Find the best baseline by PQ (excluding Ours AND Interpolation)
    baselines = [m for m in method_order
                 if m != ours_name and m != 'Interpolation']
    best_baseline = None
    best_pq = -1
    for bl in baselines:
        bl_vals = data[data['Method'] == bl]['PQ'].values
        if len(bl_vals) > 0:
            m = bl_vals.mean()
            if m > best_pq:
                best_pq = m
                best_baseline = bl

    if best_baseline is None:
        return

    vals_a = data[data['Method'] == ours_name][metric].values
    vals_b = data[data['Method'] == best_baseline][metric].values
    n = min(len(vals_a), len(vals_b))
    if n < 5:
        return
    stat, p = stats.wilcoxon(vals_a[:n], vals_b[:n], alternative='greater')

    if p < 0.001:
        sig_text = '***'
    elif p < 0.01:
        sig_text = '**'
    elif p < 0.05:
        sig_text = '*'
    else:
        sig_text = 'n.s.'

    # Position the bar
    y_max = max(data[data['Method'].isin([ours_name, best_baseline])][metric].max(),
                0.01)
    y_bar = y_max * 1.05
    y_text = y_max * 1.08

    # Find x positions
    x_a = method_order.index(ours_name)
    x_b = method_order.index(best_baseline)

    ax.plot([x_b, x_b, x_a, x_a],
            [y_bar - 0.005, y_bar, y_bar, y_bar - 0.005],
            color='black', linewidth=0.6)
    ax.text((x_a + x_b) / 2, y_text,
            sig_text,
            ha='center', va='bottom', fontsize=6, fontweight='bold')


def plot_main_panel(df_plot, vis_data, mask_cache_path,
                    match_mode, match_threshold, condition_desc,
                    vis_sample_idx=3):
    """Plot Figure_4_Panel_a.pdf: left=visual, right=PQ+Dice violins."""
    fig = plt.figure(figsize=(180 * mm, 75 * mm))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.15)

    # --- Left panel: 2 rows × 4 cols (compact) ---
    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 4, subplot_spec=gs[0],
        height_ratios=[1, 1], hspace=0.08, wspace=0.05)

    display_order = ['Interpolation', '3DRCAN', 'Ours', 'Ground Truth']
    method_key_map = {}
    for k in vis_data.keys():
        rn = rename_method(k)
        if rn in display_order:
            method_key_map[rn] = k

    z = VIS_SLICE_IDX

    # Load masks for the requested sample from cache
    with h5py.File(mask_cache_path, 'r') as mf:
        gt_mask = mf['masks/Ground_Truth'][vis_sample_idx, z]
        vis_masks = {'Ground Truth': gt_mask}
        num_methods = mf.attrs['num_methods']
        for i in range(num_methods):
            mk = mf.attrs[f'method_{i}']
            if mk == 'Ground Truth':
                continue
            safe = mk.replace(' ', '_').replace('(', '').replace(')', '')
            vis_masks[mk] = mf[f'masks/{safe}'][vis_sample_idx, z]

    for col, disp_name in enumerate(display_order):
        raw_key = method_key_map.get(disp_name, disp_name)

        ax_img = fig.add_subplot(gs_left[0, col])
        img = vis_data[raw_key][z]
        ax_img.imshow(img, cmap='gray', interpolation='nearest')
        ax_img.set_title(disp_name, fontsize=7, fontweight='bold', pad=2)
        ax_img.axis('off')

        ax_seg = fig.add_subplot(gs_left[1, col])
        if disp_name == 'Ground Truth':
            gt_rgb = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
            gt_rgb[gt_mask > 0] = COLOR_GT
            ax_seg.imshow(gt_rgb)
        else:
            mask_key = raw_key
            pred_m = vis_masks.get(mask_key, gt_mask)
            sem_rgb, tp, fp, fn = compute_semantic_mask(
                pred_m, gt_mask, match_mode, match_threshold)
            ax_seg.imshow(sem_rgb)
        ax_seg.axis('off')

    # --- Right panel: PQ (top) + Dice (bottom) violins (main 3 methods only) ---
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.45)

    main_palette = {
        'Interpolation': '#7f7f7f',
        '3DRCAN': '#1f77b4',
        'Ours': '#d62728'
    }
    main_methods = ['Interpolation', '3DRCAN', 'Ours']
    df_m = df_plot[df_plot['Method'].isin(main_methods)]

    # PQ violin (top)
    ax_pq = fig.add_subplot(gs_right[0])
    sns.violinplot(
        data=df_m, x='Method', y='PQ',
        hue='Method', order=main_methods, palette=main_palette,
        ax=ax_pq, linewidth=1.0, inner='box', cut=0)
    ax_pq.set_ylabel('Panoptic Quality', fontsize=7, labelpad=2)
    ax_pq.set_xlabel('')
    ax_pq.tick_params(axis='x', labelsize=6, length=0)
    ax_pq.tick_params(axis='y', labelsize=5)
    if ax_pq.get_legend():
        ax_pq.get_legend().remove()
    sns.despine(ax=ax_pq)
    add_significance(ax_pq, df_m, metric='PQ', method_order=main_methods)

    # Dice violin (bottom)
    ax_dice = fig.add_subplot(gs_right[1])
    sns.violinplot(
        data=df_m, x='Method', y='Dice',
        hue='Method', order=main_methods, palette=main_palette,
        ax=ax_dice, linewidth=1.0, inner='box', cut=0)
    ax_dice.set_ylabel('Dice Score', fontsize=7, labelpad=2)
    ax_dice.set_xlabel('')
    ax_dice.tick_params(axis='x', labelsize=6, length=0)
    ax_dice.tick_params(axis='y', labelsize=5)
    if ax_dice.get_legend():
        ax_dice.get_legend().remove()
    sns.despine(ax=ax_dice)
    add_significance(ax_dice, df_m, metric='Dice', method_order=main_methods)

    # Legend
    legend_elements = [
        Patch(facecolor=COLOR_GT, label='GT Reference'),
        Patch(facecolor=COLOR_TP, label='True Positive'),
        Patch(facecolor=COLOR_FP, label='False Positive'),
        Patch(facecolor=COLOR_FN, label='False Negative'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.35, -0.02),
               ncol=4, frameon=False, fontsize=5.5,
               handlelength=1.2, handletextpad=0.3, columnspacing=0.8)

    plt.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.12)
    os.makedirs('../outputs', exist_ok=True)
    out = '../outputs/Figure_4_Panel_a_new.pdf'
    plt.savefig(out, dpi=600, transparent=True)
    plt.close(fig)


def plot_supplementary(df_plot, condition_desc):
    """Plot Sup_Fig_4.pdf: F1 violin plot with ALL 5 methods."""
    fig = plt.figure(figsize=(120 * mm, 45 * mm))
    gs = fig.add_gridspec(1, 1)

    sup_palette = {
        'Ours': '#E64B35',
        '3DRCAN': '#4DBBD5',
        'RLN': '#8491B4',
        'SwinIR': '#B0B9D1',
        'Interpolation': '#E1E5ED'
    }
    sup_methods = ['Ours', '3DRCAN', 'RLN', 'SwinIR', 'Interpolation']
    df_m = df_plot[df_plot['Method'].isin(sup_methods)]

    ax = fig.add_subplot(gs[0])
    sns.violinplot(
        data=df_m, x='Method', y='F1',
        hue='Method', order=sup_methods, palette=sup_palette,
        ax=ax, linewidth=1.0, inner='box', cut=0)
    ax.set_ylabel('F1 Score', fontsize=7, labelpad=2)
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=6, length=0)
    ax.tick_params(axis='y', labelsize=5)
    if ax.get_legend():
        ax.get_legend().remove()
    sns.despine(ax=ax)
    add_significance(ax, df_m, metric='F1',
                     method_order=sup_methods)

    plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)
    os.makedirs('../outputs', exist_ok=True)
    out = '../outputs/Sup_Fig_4_new.pdf'
    plt.savefig(out, dpi=600, transparent=True)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Parse CLI args ──
    parser = argparse.ArgumentParser(description='Figure 4 Panel A')
    parser.add_argument('--vis-idx', type=int, default=429,
                        help='Sample index for visualization (default: 6)')
    parser.add_argument('--vis-slice', type=int, default=8,
                        help='Slice index (z) for visualization (default: 10)')
    args = parser.parse_args()
    VIS_SAMPLE_IDX = args.vis_idx
    VIS_SLICE_IDX = args.vis_slice

    # ── Phase 1: Run cellpose once ──
    if os.path.exists(MASK_CACHE):

        # Still need vis_data for plotting
        with h5py.File(H5_PATH, 'r') as f:
            vis_idx = VIS_SAMPLE_IDX
            with h5py.File(MASK_CACHE, 'r') as mf:
                if 'vis_sample_idx' in mf.attrs:
                    vis_idx = int(mf.attrs['vis_sample_idx'])
                # Override with CLI arg if provided
                vis_idx = VIS_SAMPLE_IDX

            hr_vol = f['hr'][vis_idx].squeeze()
            lr_raw = f['lr'][vis_idx].squeeze()
            lr_interp = interpolate_lr_volume(lr_raw, target_depth=20)
            vis_data = {
                'Ground Truth': [hr_vol[z] for z in range(20)],
                'LR (Interp)': [lr_interp[z] for z in range(20)]
            }
            output_keys = [k for k in f.keys() if "output" in k or "sr" in k]
            for k in output_keys:
                data = f[k][vis_idx].squeeze()
                if len(data.shape) == 3 and data.shape[0] == 20:
                    vis_data[k] = [data[z] for z in range(20)]
    else:
        vis_data, _ = run_cellpose_and_cache(
            H5_PATH, MASK_CACHE, diameter=DIAMETER,
            vis_sample_idx=VIS_SAMPLE_IDX)

    # ── Phase 2: Compute metrics for all conditions ──
    os.makedirs(METRICS_CACHE_DIR, exist_ok=True)
    all_results = {}

    for mode, thresh in MATCH_CONDITIONS:
        cond_key = f'{mode}_{thresh}'
        cache_path = os.path.join(METRICS_CACHE_DIR, f'{cond_key}.csv')

        if os.path.exists(cache_path):

            df = pd.read_csv(cache_path)
        else:

            df = compute_metrics_from_cache(MASK_CACHE, mode, thresh)
            df.to_csv(cache_path, index=False)

        df['Method'] = df['Method'].apply(rename_method)
        all_results[cond_key] = (mode, thresh, df)

    # ── Select fixed condition for plotting ──
    assert PLOT_CONDITION in all_results, \
        f"Condition '{PLOT_CONDITION}' not found. Available: {list(all_results.keys())}"
    plot_mode, plot_thresh, df_plot = all_results[PLOT_CONDITION]
    plot_desc = f'IoU > {plot_thresh}' if plot_mode == 'iou' \
        else f'Centroid < {plot_thresh} px'

    # ── Generate figures ──
    plot_main_panel(df_plot, vis_data, MASK_CACHE,
                    plot_mode, plot_thresh, plot_desc,
                    vis_sample_idx=VIS_SAMPLE_IDX)
    plot_supplementary(df_plot, plot_desc)
