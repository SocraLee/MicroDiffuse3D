import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import nature_style
import h5py
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from scipy import stats
from matplotlib.lines import Line2D

# --- 1. 配置 ---
mm = nature_style.apply_nature_style()

# --- 2. 核心：带缓存的数据管道 ---
cache_file = 'fig4_cache.csv'
read_cache = True


def computer_metrics(hr, lr, baseline, ours) -> pd.DataFrame:
    hr_d = torch.diagonal(hr, dim1=-2, dim2=-1).numpy()
    lr_d = torch.diagonal(lr, dim1=-2, dim2=-1).numpy()
    base_d = torch.diagonal(baseline, dim1=-2, dim2=-1).numpy()
    ours_d = torch.diagonal(ours, dim1=-2, dim2=-1).numpy()

    N, Z, L = hr_d.shape
    records = []

    def calc_1d_metrics(pred, target):
        pcc, _ = pearsonr(pred, target)
        mean_p, mean_t = np.mean(pred), np.mean(target)
        var_p, var_t = np.var(pred), np.var(target)
        cov = np.mean((pred - mean_p) * (target - mean_t))
        ccc = (2 * cov) / (var_p + var_t + (mean_p - mean_t) ** 2)
        rmse = np.sqrt(np.mean((pred - target) ** 2))
        drange = np.max(target) - np.min(target) + 1e-8
        nrmse = rmse / drange
        ws = min(7, len(target))
        if ws % 2 == 0: ws -= 1
        ssim_val = ssim(target, pred, data_range=drange, win_size=ws)
        return pcc, ccc, nrmse, ssim_val

    for i in tqdm(range(N), desc="Computing 1D Metrics"):
        for z in range(Z):
            gt_val = hr_d[i, z]
            p_in, c_in, n_in, s_in = calc_1d_metrics(lr_d[i, z], gt_val)
            records.append({'Sample_Idx': i, 'Slice_Idx': z,
                            'Method': 'Interpolation', 'PCC': p_in, 'CCC': c_in,
                            'N-RMSE': n_in, '1D-SSIM': s_in, "Input_CCC": c_in})
            p_b, c_b, n_b, s_b = calc_1d_metrics(base_d[i, z], gt_val)
            records.append({'Sample_Idx': i, 'Slice_Idx': z,
                            'Method': '3DRCAN', 'PCC': p_b, 'CCC': c_b,
                            'N-RMSE': n_b, '1D-SSIM': s_b, 'Input_CCC': c_in})
            p_o, c_o, n_o, s_o = calc_1d_metrics(ours_d[i, z], gt_val)
            records.append({'Sample_Idx': i, 'Slice_Idx': z,
                            'Method': 'Ours', 'PCC': p_o, 'CCC': c_o,
                            'N-RMSE': n_o, '1D-SSIM': s_o, 'Input_CCC': c_in})

    return pd.DataFrame(records)


# 3DSR HDF5 with `hr`, `lr`, `microdiffuse3d`, and `RCAN_output` model prediction datasets.
data_path = '<YOUR_DATA_PATH>'

if os.path.exists(cache_file) and read_cache:
    df = pd.read_csv(cache_file)
    data = h5py.File(data_path, 'r')
    hr = torch.from_numpy(data['hr'][:]).squeeze()
    lr = torch.from_numpy(data['lr'][:]).squeeze()
    sit = torch.from_numpy(data['microdiffuse3d'][:]).squeeze()
    rcan = torch.from_numpy(data['RCAN_output'][:]).squeeze()
    if hr.shape != lr.shape:
        inputs = lr.unsqueeze(1)
        lr = F.interpolate(inputs, size=(hr.shape[1], 256, 256),
                           mode='trilinear', align_corners=False).squeeze(1)
else:
    data = h5py.File(data_path, 'r')
    hr = torch.from_numpy(data['hr'][:]).squeeze()
    lr = torch.from_numpy(data['lr'][:]).squeeze()
    sit = torch.from_numpy(data['microdiffuse3d'][:]).squeeze()
    rcan = torch.from_numpy(data['RCAN_output'][:]).squeeze()
    if hr.shape != lr.shape:
        inputs = lr.unsqueeze(1)
        lr = F.interpolate(inputs, size=(hr.shape[1], 256, 256),
                           mode='trilinear', align_corners=False).squeeze(1)
    df = computer_metrics(hr, lr, rcan, sit)
    df.to_csv(cache_file, index=False)




# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
palette = {
    'Interpolation': '#7f7f7f',
    '3DRCAN': '#1f77b4',
    'Ours': '#d62728'
}
method_order = ['Interpolation', '3DRCAN', 'Ours']


def add_significance(ax, data, metric, method_order_list,
                     method_a='Ours', method_b='3DRCAN'):
    """Add Wilcoxon significance bar between method_a and method_b."""
    vals_a = data[data['Method'] == method_a][metric].values
    vals_b = data[data['Method'] == method_b][metric].values
    n = min(len(vals_a), len(vals_b))
    if n < 5:
        return
    if 'RMSE' in metric:
        stat, p = stats.wilcoxon(vals_a[:n], vals_b[:n], alternative='less')
    else:
        stat, p = stats.wilcoxon(vals_a[:n], vals_b[:n], alternative='greater')

    if p < 0.001:
        sig_text = '***'
    elif p < 0.01:
        sig_text = '**'
    elif p < 0.05:
        sig_text = '*'
    else:
        sig_text = 'n.s.'

    y_max = data[data['Method'].isin([method_a, method_b])][metric].max()
    y_bar = y_max * 1.10
    y_text = y_max * 1.14
    x_a = method_order_list.index(method_a)
    x_b = method_order_list.index(method_b)
    ax.plot([x_b, x_b, x_a, x_a],
            [y_bar - 0.005, y_bar, y_bar, y_bar - 0.005],
            color='black', linewidth=0.6)
    ax.text((x_a + x_b) / 2, y_text, sig_text,
            ha='center', va='bottom', fontsize=6, fontweight='bold')


os.makedirs('../outputs', exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Main Figure: Figure_4_Panel_cd.pdf
# Left: images + 1D profile, Right: regression + CCC violin
# ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(180 * mm, 85 * mm))
gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.15)

gs_left = gridspec.GridSpecFromSubplotSpec(
    2, 4, subplot_spec=gs[0], height_ratios=[1, 1], hspace=0.2, wspace=0.05)

# sample_idx, slice_idx = 1180, 10
sample_idx, slice_idx = 1180, 10
img_lr = lr[sample_idx, slice_idx].numpy()
img_base = rcan[sample_idx, slice_idx].numpy()
img_ours = sit[sample_idx, slice_idx].numpy()
img_gt = hr[sample_idx, slice_idx].numpy()

imgs = [img_lr, img_base, img_ours, img_gt]
titles = ['Interpolation', '3DRCAN', 'Ours', 'Ground Truth']

for i in range(4):
    ax = fig.add_subplot(gs_left[0, i])
    ax.imshow(imgs[i], cmap='gray')
    ax.set_title(titles[i], fontsize=8, fontweight='bold')
    ax.axis('off')
    ax.plot([0, 255], [0, 255], color='#2ca02c', linewidth=1.2, alpha=0.9)

ax_prof = fig.add_subplot(gs_left[1, :])
x_axis = np.arange(256)
ax_prof.plot(x_axis, np.diag(img_lr), color='gray', linestyle=':',
             linewidth=1, label='Interpolation', alpha=0.8)
ax_prof.plot(x_axis, np.diag(img_base), color='#1f77b4', linestyle='--',
             linewidth=1.2, label='3DRCAN')
ax_prof.plot(x_axis, np.diag(img_ours), color='#d62728', linestyle='-',
             linewidth=1.2, label='Ours')
ax_prof.plot(x_axis, np.diag(img_gt), color='black', linestyle='-',
             linewidth=1.2, label='Ground Truth')
ax_prof.set_xlabel(r'Position along diagonal ($\mu$m)', fontsize=7)
ax_prof.set_ylabel('Normalized Intensity', fontsize=7)
# Relabel x-axis ticks from pixels to µm (512 px = 301.176 µm, diagonal = px * √2 * scale)
um_per_px = 301.176 / 512.0
diag_scale = um_per_px * np.sqrt(2)  # µm per diagonal-pixel
# Pick round µm values, then back-calculate pixel positions for tick placement
tick_um = np.array([0, 40, 80, 120, 160, 200])
tick_pixels = tick_um / diag_scale
ax_prof.set_xticks(tick_pixels)
ax_prof.set_xticklabels([f'{v:.0f}' for v in tick_um])
ax_prof.tick_params(axis='both', which='major', labelsize=6)
ax_prof.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
               ncol=4, frameon=False, fontsize=7)
sns.despine(ax=ax_prof)

# Right panel: regression (top) + CCC violin (bottom)
gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.4)

# Regression plot
ax_reg = fig.add_subplot(gs_right[0])
df_base = df[df['Method'] == '3DRCAN']
df_ours_reg = df[df['Method'] == 'Ours']

min_val = min(df['Input_CCC'].min(), df['CCC'].min())
max_val = max(df['Input_CCC'].max(), df['CCC'].max())
ax_reg.plot([min_val, max_val], [min_val, max_val],
            color='gray', linestyle='--', linewidth=1)

sns.regplot(data=df_base, x='Input_CCC', y='CCC', ax=ax_reg,
            color=palette['3DRCAN'], scatter=False, ci=None,
            line_kws={'linewidth': 1.5}, order=4)
sns.regplot(data=df_ours_reg, x='Input_CCC', y='CCC', ax=ax_reg,
            color=palette['Ours'], scatter=False, ci=None,
            line_kws={'linewidth': 1.5}, order=4)

ax_reg.set_title('Signal Concordance over Input Quality', fontsize=8,
                 fontweight='bold')
ax_reg.set_xlabel('Input-to-GT CCC', fontsize=7)
ax_reg.set_ylabel('Model-to-GT CCC', fontsize=7)
ax_reg.tick_params(axis='both', labelsize=6)
sns.despine(ax=ax_reg)

legend_handles = [
    Line2D([0], [0], color='gray', linestyle='--', linewidth=1,
           label='y=x (No Improvement)'),
    Line2D([0], [0], color=palette['3DRCAN'], linewidth=1.5,
           label='3DRCAN'),
    Line2D([0], [0], color=palette['Ours'], linewidth=1.5,
           label='Ours'),
]
ax_reg.legend(handles=legend_handles, loc='upper left',
              frameon=False, fontsize=6)

# CCC violin (with significance)
ax_ccc = fig.add_subplot(gs_right[1])
sns.violinplot(data=df, x='Method', y='CCC', ax=ax_ccc, hue='Method',
               order=method_order, palette=palette,
               linewidth=1.0, inner='box', cut=0)
ax_ccc.set_xlabel('')
ax_ccc.set_ylabel('Signal Concordance (CCC)', fontsize=7)
ax_ccc.tick_params(axis='both', labelsize=6)
if ax_ccc.get_legend():
    ax_ccc.get_legend().remove()
sns.despine(ax=ax_ccc)
# Significance bar: position based on violin visual top, not data max
y_lo, y_hi = ax_ccc.get_ylim()
vals_a = df[df['Method'] == 'Ours']['CCC'].values
vals_b = df[df['Method'] == '3DRCAN']['CCC'].values
n = min(len(vals_a), len(vals_b))
stat, p = stats.wilcoxon(vals_a[:n], vals_b[:n], alternative='greater')
sig_text = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
y_bar = y_hi + (y_hi - y_lo) * 0.03
y_text = y_bar + (y_hi - y_lo) * 0.02
x_a = method_order.index('Ours')
x_b = method_order.index('3DRCAN')
ax_ccc.plot([x_b, x_b, x_a, x_a],
            [y_bar - 0.005, y_bar, y_bar, y_bar - 0.005],
            color='black', linewidth=0.6)
ax_ccc.text((x_a + x_b) / 2, y_text, sig_text,
            ha='center', va='bottom', fontsize=6, fontweight='bold')
ax_ccc.set_ylim(y_lo, y_text + (y_hi - y_lo) * 0.10)

plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.20)
plt.savefig('../outputs/Figure_4_Panel_cd.pdf', dpi=600, transparent=True)
plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Sup_Fig_5.pdf: PCC + 1D-SSIM violins (2 rows)
# ──────────────────────────────────────────────────────────────
fig5 = plt.figure(figsize=(90 * mm, 70 * mm))
gs5 = fig5.add_gridspec(2, 1, hspace=0.45)

metric_list = [
    ('PCC',     'Pearson Correlation (PCC)'),
    ('1D-SSIM', '1D Structural Similarity'),
]

for i, (metric, ylabel) in enumerate(metric_list):
    ax = fig5.add_subplot(gs5[i])
    sns.violinplot(data=df, x='Method', y=metric, ax=ax,
                   hue='Method', order=method_order, palette=palette,
                   linewidth=0.5, inner='box', cut=0)
    ax.set_ylabel(ylabel, fontsize=7, labelpad=2)
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=6, length=0)
    ax.tick_params(axis='y', labelsize=5)
    if ax.get_legend():
        ax.get_legend().remove()
    sns.despine(ax=ax)
    add_significance(ax, df, metric, method_order)

plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.06)
plt.savefig('../outputs/Sup_Fig_5.pdf', dpi=600, transparent=True)
plt.close(fig5)
