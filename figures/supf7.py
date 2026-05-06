"""
Sup Fig 7 — Frequency Analysis & Perception-Distortion Tradeoff
Row 1: Lateral (X-Y) Radial Power Spectrum  (Ours, 3DRCAN, SwinIR, GT)
Row 2: Axial  (X-Z) Radial Power Spectrum   (same)
Row 3: Lateral P-D | Axial P-D              (all methods incl. RLN, CARE, SRCNN)

Usage: python supf7.py --dataset {3DSR,Denoise,BioTISR}
"""
import os, pickle, argparse
from collections import OrderedDict
import numpy as np
import torch, torch.nn.functional as TF
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
import nature_style
from tqdm import tqdm

# Per-dataset HDF5 result files. Each must contain `hr` (ground truth), `lr`
# (input), and the model prediction datasets keyed by the entries in
# RPS_METHODS / PD_EXTRA below.
DATASETS = {
    '3DSR':    '<YOUR_DATA_PATH>',
    'Denoise': '<YOUR_DATA_PATH>',
    'BioTISR': '<YOUR_DATA_PATH>',
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Denoise', choices=DATASETS.keys(),
                    help='Dataset name (determines H5 path and output name)')
parser.add_argument('--idx', type=int, default=23,
                    help='Index of the volume to visualize in panel c')
args = parser.parse_args()

H5_PATH = DATASETS[args.dataset]
CACHE_FILE = f'supf7_fft_cache_{args.dataset}.pkl'
OUT_PDF = f'../outputs/Sup_Fig_7_{args.dataset}.pdf'
HF_CUTOFF = 0.375  # cycles/pixel — top 25% of Nyquist (0.5*0.75)
mm = nature_style.apply_nature_style()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Methods shown in RPS (spectral) plots
RPS_METHODS = OrderedDict([
    ('microdiffuse3d', 'Ours'),
    ('RCAN_output',              '3DRCAN'),
    ('SwinIR_output',            'SwinIR'),
])
# Extra methods for P-D scatter only
PD_EXTRA = OrderedDict([
    ('RLN_output',   'RLN'),
    ('CARE_output',  'CARE'),
    ('SRCNN_output', 'SRCNN'),
])
ALL_METHODS = OrderedDict(list(RPS_METHODS.items()) + list(PD_EXTRA.items()))

PALETTE = {
    'Ground Truth': '#2D2D2D',
    'Ours':         '#E64B35',
    '3DRCAN':       '#F39B7F',
    'SwinIR':       '#4DBBD5',
    'RLN':          '#E8A838',
    'CARE':         '#8491B4',
    'SRCNN':        '#B0B9D1',
}
MARKER = {
    'Ours': 'o', '3DRCAN': 's', 'SwinIR': 'D',
    'RLN': '^', 'CARE': 'v', 'SRCNN': 'P',
}
LSTYLE = {'Ground Truth': '-', 'Ours': '-', '3DRCAN': '--', 'SwinIR': ':'}

# ── Helpers ──
def radial_average(p):
    H, W = p.shape; cy, cx = H//2, W//2; mr = min(cy, cx)
    y = torch.arange(H, device=p.device).float() - cy
    x = torch.arange(W, device=p.device).float() - cx
    Y, X = torch.meshgrid(y, x, indexing='ij')
    R = torch.sqrt(X**2+Y**2).long().clamp(max=mr-1)
    s = torch.zeros(mr, device=p.device, dtype=torch.float64)
    c = torch.zeros(mr, device=p.device, dtype=torch.float64)
    s.scatter_add_(0, R.flatten(), p.double().flatten())
    c.scatter_add_(0, R.flatten(), torch.ones(H*W, device=p.device, dtype=torch.float64))
    return (s/c.clamp(min=1)).cpu().numpy()

@torch.no_grad()
def analyze_fft(f, key, N):
    """Expensive FFT computation — results are cached."""
    ls = axs = None
    rmse_samples = []
    lat_radials, ax_radials = [], []   # per-sample radial profiles
    for i in tqdm(range(N), leave=False):
        raw = torch.from_numpy(f[key][i]).squeeze().float().to(device)
        gt  = torch.from_numpy(f['hr'][i]).squeeze().float().to(device)
        D, H, W = raw.shape
        Fl = torch.fft.fftshift(torch.fft.fft2(raw), dim=(-2,-1))
        Pl = (Fl.abs()**2).mean(0)
        if ls is None: ls = torch.zeros_like(Pl, dtype=torch.float64)
        ls += Pl.double()
        lat_radials.append(radial_average(Pl))
        Fa = torch.fft.fftshift(torch.fft.fft2(raw.permute(1,0,2)), dim=(-2,-1))
        Pa = (Fa.abs()**2).mean(0)
        if axs is None: axs = torch.zeros_like(Pa, dtype=torch.float64)
        axs += Pa.double()
        ax_radials.append(radial_average(Pa))
        rmse_samples.append(torch.sqrt(((raw-gt)**2).mean()).item())
    return {
        'lateral_radial': radial_average((ls/N).float()),
        'axial_radial':   radial_average((axs/N).float()),
        'rmse_samples': rmse_samples,
        'lat_radials': lat_radials,
        'ax_radials':  ax_radials,
    }

# ── Phase 1: FFT cache (cutoff-independent) ──
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as fp: fft_cache = pickle.load(fp)
else:
    fft_cache = {}
    with h5py.File(H5_PATH, 'r') as f:
        N = f['hr'].shape[0]; avail = set(f.keys())
        fft_cache['Ground Truth'] = analyze_fft(f, 'hr', N)
        for k, nm in ALL_METHODS.items():
            if k not in avail: continue
            fft_cache[nm] = analyze_fft(f, k, N)
    with open(CACHE_FILE, 'wb') as fp: pickle.dump(fft_cache, fp)

# ── Phase 1b: Derive HF metrics from cached radial profiles (cheap) ──
def compute_hf_metrics(fft_cache, cutoff):
    gt = fft_cache['Ground Truth']
    out = {}
    for name, r in fft_cache.items():
        N = len(r['rmse_samples'])
        hl, ha, hfpow_l, hfpow_a = [], [], [], []
        for i in range(N):
            for plane, (pr, gt_r, h_list, hp_list) in [
                ('lat', (r['lat_radials'][i], gt['lat_radials'][i], hl, hfpow_l)),
                ('ax',  (r['ax_radials'][i],  gt['ax_radials'][i],  ha, hfpow_a)),
            ]:
                mr = len(pr)
                freq = np.arange(mr) / (2*mr)
                hf = freq > cutoff
                if not hf.any(): continue
                h_list.append(np.abs(pr[hf] - gt_r[hf]).mean())
                hp_list.append(pr[hf].mean())
        rmse_arr = np.array(r['rmse_samples'])
        out[name] = {
            'lateral_radial': r['lateral_radial'],
            'axial_radial':   r['axial_radial'],
            'rmse':       (float(rmse_arr.mean()), float(rmse_arr.std())),
            'hf_lat':     (float(np.mean(hl)),     float(np.std(hl))),
            'hf_ax':      (float(np.mean(ha)),     float(np.std(ha))),
            'hfpow_lat':  (float(np.mean(hfpow_l)), float(np.std(hfpow_l))),
            'hfpow_ax':   (float(np.mean(hfpow_a)), float(np.std(hfpow_a))),
        }
    return out

results = compute_hf_metrics(fft_cache, HF_CUTOFF)


# ── Phase 2: Plot ──
rps_order = ['3DRCAN', 'SwinIR', 'RLN', 'CARE', 'SRCNN', 'Ours']  # Ours last (on top)
pd_order  = ['3DRCAN', 'SwinIR', 'RLN', 'CARE', 'SRCNN', 'Ours']  # Ours last (on top)

fig = plt.figure(figsize=(90*mm, 130*mm))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1.2], hspace=0.8,
                       left=0.14, right=0.95, top=0.92, bottom=0.05)

ZORDER_RPS = {'Ours': 6, '3DRCAN': 5, 'SwinIR': 4, 'RLN': 3, 'CARE': 2, 'SRCNN': 1}

def plot_rps(ax, key):
    for n in rps_order:
        if n not in results: continue
        rps = results[n][key]; freq = np.arange(len(rps))/(2*len(rps))
        hf = freq >= HF_CUTOFF
        freq_norm = freq[hf] / 0.5
        ax.plot(freq_norm, rps[hf],
                color=PALETTE[n], linestyle=LSTYLE.get(n,'-'),
                linewidth=1.4 if n == 'Ours' else 0.9, alpha=0.95,
                zorder=ZORDER_RPS.get(n, 1))
    ax.set_xlabel('Normalized Frequency (Top 25%)', fontsize=7)
    ax.set_ylabel('Power', fontsize=7)
    ax.tick_params(labelsize=7); sns.despine(ax=ax)

def plot_pd(ax, hf_key):
    for n in pd_order:
        if n not in results: continue
        r = results[n]
        ax.errorbar(r['rmse'][0], r[hf_key][0],
                    xerr=r['rmse'][1], yerr=r[hf_key][1],
                    fmt=MARKER.get(n, 'o'), color=PALETTE[n],
                    markersize=4, markeredgecolor='black', markeredgewidth=0.3,
                    capsize=1.5, capthick=0.4, elinewidth=0.4,
                    zorder=5, label=n)
    ax.plot(0, 0, marker='*', color=PALETTE['Ground Truth'],
            markersize=7, markeredgecolor='black', markeredgewidth=0.3,
            zorder=10, linestyle='none', label='Ground Truth')
    ax.set_xlabel('RMSE', fontsize=7)
    ax.set_ylabel('HF Spectral Error', fontsize=7)
    ax.tick_params(labelsize=7); sns.despine(ax=ax)

ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])

plot_rps(ax_a, 'lateral_radial')
plot_pd(ax_b,  'hf_lat')

# RPS legend (panel a top)
rps_legend_order = ['Ours', '3DRCAN', 'RLN', 'SwinIR', 'CARE', 'SRCNN']
h_rps = [Line2D([0],[0], color=PALETTE[n], linestyle=LSTYLE.get(n,'-'),
                linewidth=1.2, label=n) for n in rps_legend_order if n in results]
leg_a = ax_a.legend(handles=h_rps, loc='lower center', bbox_to_anchor=(0.5, 1.05),
                    ncol=len(h_rps), frameon=False, fontsize=7,
                    handlelength=1.5, handletextpad=0.3, columnspacing=1.0)

# P-D legend (panel b top, between the two panels)
pd_legend_order = ['Ours', '3DRCAN', 'RLN', 'SwinIR', 'CARE', 'SRCNN', 'Ground Truth']
h_pd = []
for n in pd_legend_order:
    if n not in results and n != 'Ground Truth': continue
    m = '*' if n == 'Ground Truth' else MARKER.get(n, 'o')
    h_pd.append(Line2D([0],[0], marker=m, color=PALETTE[n], linestyle='none',
                       markersize=4, markeredgecolor='black',
                       markeredgewidth=0.3, label=n))
leg_b = ax_b.legend(handles=h_pd, loc='lower center', bbox_to_anchor=(0.54, 1.15),
                    ncol=len(h_pd), frameon=False, fontsize=7,
                    handlelength=1.0, handletextpad=0.3, columnspacing=0.8)

# Panel labels — placed relative to axes to avoid overlap when adjusting hspace
import matplotlib.transforms as mtransforms
trans_a = mtransforms.blended_transform_factory(fig.transFigure, ax_a.transAxes)
ax_a.text(0.02, 1.15, 'a', transform=trans_a, fontsize=9, fontweight='bold', va='bottom')
trans_b = mtransforms.blended_transform_factory(fig.transFigure, ax_b.transAxes)
ax_b.text(0.02, 1.15, 'b', transform=trans_b, fontsize=9, fontweight='bold', va='bottom')

# ── Phase 3: Visual Comparison (Panel c) ──
gs_c = gs[2].subgridspec(1, 5, wspace=0.05)
img_names = ['Inputs', 'CARE', '3DRCAN', 'Ours', 'Target']

imgs_c = {}
with h5py.File(H5_PATH, 'r') as f:
    idx_target = args.idx
    z_target = 10
    if 'lr' in f:
        imgs_c['Inputs'] = np.array(f['lr'][idx_target]).squeeze()[z_target]
    elif 'hr' in f: imgs_c['Inputs'] = np.array(f['hr'][idx_target]).squeeze()[z_target]
    
    if 'hr' in f:
        imgs_c['Target'] = np.array(f['hr'][idx_target]).squeeze()[z_target]
    if 'CARE_output' in f:
        imgs_c['CARE'] = np.array(f['CARE_output'][idx_target]).squeeze()[z_target]
    elif 'hr' in f: imgs_c['CARE'] = np.array(f['hr'][idx_target]).squeeze()[z_target]
    
    if 'RCAN_output' in f:
        imgs_c['3DRCAN'] = np.array(f['RCAN_output'][idx_target]).squeeze()[z_target]
    elif 'hr' in f: imgs_c['3DRCAN'] = np.array(f['hr'][idx_target]).squeeze()[z_target]
    
    if 'microdiffuse3d' in f:
        imgs_c['Ours'] = np.array(f['microdiffuse3d'][idx_target]).squeeze()[z_target]
    elif 'hr' in f: imgs_c['Ours'] = np.array(f['hr'][idx_target]).squeeze()[z_target]

for i, name in enumerate(img_names):
    ax_c = fig.add_subplot(gs_c[0, i])
    
    # Translate panel c upwards to reduce spacing from b
    pos = ax_c.get_position()
    pos.y0 += 0.1
    pos.y1 += 0.1
    ax_c.set_position(pos)
    
    if i == 0:
        trans_c = mtransforms.blended_transform_factory(fig.transFigure, ax_c.transAxes)
        ax_c.text(0.02, 1.15, 'c', transform=trans_c, fontsize=9, fontweight='bold', va='bottom')
        
    if name in imgs_c:
        ax_c.imshow(imgs_c[name], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax_c.axis('off')
    ax_c.set_title(name, fontsize=7, pad=3)
    


os.makedirs('../outputs', exist_ok=True)
plt.savefig(OUT_PDF, dpi=600, transparent=True)
plt.close(fig)
