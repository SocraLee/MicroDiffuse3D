"""
Sup_Fig_5_ablation.pdf — Decoder Ablation: violin plot comparison
Methods:
  - MicroDiffuse3D + adapted decoder  (microdiffuse3d)
  - MicroDiffuse3D + VAE decoder      (microdiffuse3d_VAEdecoder)
  - 3DRCAN                            (RCAN)
Dataset: 3DSR
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import h5py
import nature_style

# ── 1. Style ──
mm = nature_style.apply_nature_style()

# ── 2. Data loading ──
# Per-sample metrics HDF5: keyed by `{method}_{metric}` arrays produced
# from the model prediction files (see methods dict below for expected keys).
h5_path = '<YOUR_DATA_PATH>'

# HDF5 key prefix → display name
methods = {
    'microdiffuse3d':            'Ours w/ adapted decoder',
    'microdiffuse3d_VAEdecoder': 'Ours w/ VAE decoder',
    'RCAN':                      '3DRCAN',
}

metrics_map = {
    'psnr':   'PSNR',
    'ssim':   'SSIM',
    'msssim': 'MS-SSIM',
    'lpips':  'LPIPS',
}

rows = []
with h5py.File(h5_path, 'r') as f:
    for h5_key, display_name in methods.items():
        for metric_key, metric_label in metrics_map.items():
            key = f'{h5_key}_{metric_key}'
            values = f[key][:]
            for v in values:
                rows.append({
                    'Method': display_name,
                    'Metric': metric_label,
                    'Value': float(v),
                })

df = pd.DataFrame(rows)


# ── 4. Plot ──
method_order = list(methods.values())

palette = {
    'Ours w/ adapted decoder':  '#E64B35',   # Bold red — best
    'Ours w/ VAE decoder':      '#F39B7F',   # Salmon — ablation
    '3DRCAN':                   '#4DBBD5',   # Teal — baseline
}

metric_list = ['PSNR', 'SSIM', 'MS-SSIM', 'LPIPS']

fig = plt.figure(figsize=(180 * mm, 50 * mm))
gs = fig.add_gridspec(1, 4, wspace=0.35)


for i, metric_name in enumerate(metric_list):
    ax = fig.add_subplot(gs[0, i])
    subset = df[df['Metric'] == metric_name]

    sns.violinplot(
        data=subset, x='Method', y='Value', hue='Method',
        order=method_order, palette=palette,
        ax=ax, linewidth=0.5, inner='box', cut=0)

    ax.set_ylabel(metric_name, fontsize=7, labelpad=2)
    ax.set_xlabel('')
    ax.set_xticklabels([])          # hide per-subplot x labels
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', labelsize=7)
    if ax.get_legend():
        ax.get_legend().remove()
    sns.despine(ax=ax)

# Shared legend at top
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=palette[m], edgecolor='black', linewidth=0.7, label=m)
                  for m in method_order]
fig.legend(handles=legend_handles, loc='upper center',
           bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
           fontsize=7, handlelength=1.2, handletextpad=0.4, columnspacing=1.5)

plt.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.06)
plt.savefig('../outputs/Sup_Fig_6.pdf', dpi=600, transparent=True)
plt.close(fig)
