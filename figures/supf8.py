"""
Sup Fig 8 — Laplacian Variance (Sharpness) Comparison
Per-sample Laplacian variance for Ours, 3DRCAN, and HR (Ground Truth).

Usage: python supf8.py --dataset {3DSR,Denoise,BioTISR}
"""
import os, pickle, argparse
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as TF
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import nature_style
from scipy import stats
from tqdm import tqdm

# Per-dataset HDF5 result files. Each must contain `hr` (ground truth) and
# the model prediction datasets keyed by the entries in METHODS below.
DATASETS = {
    '3DSR':    '<YOUR_DATA_PATH>',
    'Denoise': '<YOUR_DATA_PATH>',
    'BioTISR': '<YOUR_DATA_PATH>',
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='3DSR', choices=DATASETS.keys())
args = parser.parse_args()

H5_PATH    = DATASETS[args.dataset]
CACHE_FILE = f'supf8_lapvar_cache_{args.dataset}.pkl'
OUT_PDF    = f'../outputs/Sup_Fig_8_{args.dataset}.pdf'
mm = nature_style.apply_nature_style()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

METHODS = OrderedDict([
    ('microdiffuse3d', 'Ours'),
    ('RCAN_output',              '3DRCAN'),
    ('hr',                       'HR'),
])

PALETTE = {
    'HR':     '#2D2D2D',
    'Ours':   '#E64B35',
    '3DRCAN': '#F39B7F',
}

# 2D Laplacian kernel
LAPLACIAN_KERNEL = torch.tensor([[0., 1., 0.],
                                 [1., -4., 1.],
                                 [0., 1., 0.]], device=device).view(1, 1, 3, 3)

@torch.no_grad()
def laplacian_variance(vol):
    """Compute Laplacian variance of a 3D volume (D,H,W) via 2D Laplacian per slice."""
    v = vol.unsqueeze(1)  # (D,1,H,W)
    lap = TF.conv2d(v, LAPLACIAN_KERNEL, padding=1)
    return lap.var().item()

@torch.no_grad()
def analyze(f, key, N):
    vals = []
    for i in tqdm(range(N), leave=False):
        vol = torch.from_numpy(f[key][i]).squeeze().float().to(device)
        vals.append(laplacian_variance(vol))
    return np.array(vals)

# ── Phase 1: Cache per-sample Laplacian variance ──
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as fp: cache = pickle.load(fp)
else:
    cache = {}
    with h5py.File(H5_PATH, 'r') as f:
        N = f['hr'].shape[0]; avail = set(f.keys())
        for k, nm in METHODS.items():
            if k not in avail: continue
            cache[nm] = analyze(f, k, N)
    with open(CACHE_FILE, 'wb') as fp: pickle.dump(cache, fp)


# ── Phase 2: Plot ──
order = [nm for nm in ('Ours', '3DRCAN') if nm in cache]
pal = {nm: PALETTE[nm] for nm in order}

# Long-format DataFrame
df = pd.DataFrame(
    [(nm, v) for nm in order for v in cache[nm]],
    columns=['Method', 'Value'],
)

fig, ax = plt.subplots(figsize=(55*mm, 55*mm))
fig.subplots_adjust(left=0.26, right=0.96, top=0.90, bottom=0.14)

# Violin — f4pa style (inner='box', linewidth=1.0, cut=0)
sns.violinplot(
    data=df, x='Method', y='Value',
    hue='Method', order=order, palette=pal, legend=False,
    ax=ax, linewidth=1.0, inner='box', cut=0,
)

# Significance bar — Ours vs 3DRCAN, Wilcoxon (paired, one-sided "greater")
vals_a = cache[order[0]]
vals_b = cache[order[1]]
n = min(len(vals_a), len(vals_b))
stat, p = stats.wilcoxon(vals_a[:n], vals_b[:n], alternative='greater')

if   p < 1e-3: sig_text = '***'
elif p < 1e-2: sig_text = '**'
elif p < 5e-2: sig_text = '*'
else:          sig_text = 'n.s.'


y_max = df['Value'].max()
y_bar = y_max * 1.05
y_text = y_max * 1.08
x_a = order.index(order[0])
x_b = order.index(order[1])
ax.plot([x_b, x_b, x_a, x_a],
        [y_bar - y_max*0.005, y_bar, y_bar, y_bar - y_max*0.005],
        color='black', linewidth=0.6)
ax.text((x_a + x_b) / 2, y_text, sig_text,
        ha='center', va='bottom', fontsize=6, fontweight='bold')

ax.set_ylabel('Laplacian Variance', fontsize=7, labelpad=2)
ax.set_xlabel('')
ax.tick_params(axis='x', length=0)
sns.despine(ax=ax)

os.makedirs('../outputs', exist_ok=True)
plt.savefig(OUT_PDF, dpi=600, transparent=True)
plt.close(fig)
