"""Figure 3 Panels a,b — Denoising task: example images and metric box plots."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import h5py
import nature_style
from utils import build_long_dataframe

def read_raw_image(h5path, idx, lr_slice_idx, hr_slice_idx):
    with h5py.File(h5path, 'r') as hf:
        input_image = hf['lr'][idx]
        gt_image = hf['hr'][idx]
        input_image = input_image[lr_slice_idx, 0]
        gt_image = gt_image[hr_slice_idx, 0]
    return input_image, gt_image

mm = nature_style.apply_nature_style()

method_order = ['Ours', '3DRCAN', 'RLN', 'SwinIR', 'CARE', 'SRCNN', 'Inputs']
palette_dict = {
    'Ours':     '#E64B35',
    '3DRCAN':   '#F39B7F',
    'RLN':      '#E8A838',
    'SwinIR':   '#3C5488',
    'CARE':     '#8491B4',
    'SRCNN':    '#B0B9D1',
    'Inputs':   '#D9D9D9',
}

# Per-sample metrics HDF5 produced from model prediction files; keyed by
# `{method}_{metric}` arrays (see utils.build_long_dataframe).
file_name = '<YOUR_DATA_PATH>'
df = build_long_dataframe(file_name, "")

# Figure layout
fig = plt.figure(figsize=(180 * mm, 40 * mm))
gs_master = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.15)
gs_left = gs_master[0, 0]

# Panel a: example image pair
# Denoise HDF5 with `hr`, `lr`, and per-method model prediction datasets.
lr_raw, hr_raw = read_raw_image(
    "<YOUR_DATA_PATH>", 119, 10, 10)
name, img_lr, img_hr = ("3D Denoise", lr_raw, hr_raw)
gs_inner = gs_left.subgridspec(1, 2, wspace=0.05)
ax_lr = fig.add_subplot(gs_inner[0])
ax_hr = fig.add_subplot(gs_inner[1])
ax_lr.imshow(img_lr, cmap='gray', interpolation='nearest')
ax_hr.imshow(img_hr, cmap='gray', interpolation='nearest')
ax_lr.axis('off')
ax_hr.axis('off')
ax_lr.text(-0.1, 0.5, name, transform=ax_lr.transAxes,
           rotation=90, va='center', ha='right',
           fontsize=7, fontweight='bold', color='black')
ax_lr.set_title("Low-Quality", fontsize=7, pad=3)
ax_hr.set_title("High-Quality", fontsize=7, pad=3)

# Panel b: box plots (1x4 grid)
gs_right = gs_master[0, 1].subgridspec(1, 4, wspace=0.3)
axes_metrics = []
flat_metrics_list = ['PSNR', 'SSIM', 'MS-SSIM', 'LPIPS']

for i, metric_name in enumerate(flat_metrics_list):
    ax = fig.add_subplot(gs_right[0, i])
    axes_metrics.append(ax)

    subset = df[df['Metric'] == metric_name]
    sns.boxplot(
        data=subset, x='Dataset', y='Value', hue='Method',
        hue_order=method_order, palette=palette_dict,
        ax=ax, linecolor='black', linewidth=0.7,
        flierprops={'marker': 'o', 'markerfacecolor': 'none',
                    "markeredgecolor": "gray", "markersize": 1,
                    'markeredgewidth': 0.4, "alpha": 0.6},
        width=0.7,
    )

    ax.set_ylabel(metric_name, labelpad=2)
    ax.set_xlabel('')
    ax.tick_params(axis='x', length=0)
    ax.get_legend().remove()
    sns.despine(ax=ax)

    if i == 0:
        ax.text(-0.25, 1.05, 'b', transform=ax.transAxes, fontsize=7,
                fontweight='bold', va='bottom', ha='right')

# Shared legend
handles, labels = axes_metrics[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.67, 0.98),
           ncol=7, frameon=False, columnspacing=0.6, handletextpad=0.2, fontsize=5.5)

plt.subplots_adjust(left=0.05, right=0.98, top=0.80, bottom=0.15)
plt.savefig('../outputs/Figure_3_Panel_ab_new.pdf', dpi=600, transparent=True)
plt.show()