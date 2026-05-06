"""Figure 2 Panel c — Qualitative Z-slice comparison with X-Z orthogonal views."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import ConnectionPatch
import nature_style
import h5py
import torch
import torch.nn.functional as F

mm = nature_style.apply_nature_style()
plt.rcParams['mathtext.default'] = 'regular'


def get_image(idx=124):
    # 3DSR HDF5 with `hr`, `lr`, `microdiffuse3d`, `RCAN_output` model prediction datasets.
    data_dir = '<YOUR_DATA_PATH>'
    data = h5py.File(data_dir, 'r')
    hr = torch.from_numpy(data['hr'][idx]).squeeze()
    lr = torch.from_numpy(data['lr'][idx]).squeeze()
    sit = torch.from_numpy(data['microdiffuse3d'][idx]).squeeze()
    swinir = torch.from_numpy(data['RCAN_output'][idx]).squeeze()
    if hr.shape != lr.shape:
        with torch.no_grad():
            inputs_for_interp = lr.unsqueeze(0).unsqueeze(0)
            lr = F.interpolate(
                inputs_for_interp,
                size=(hr.shape[0], 256, 256),
                mode='trilinear', align_corners=False
            ).squeeze()
    gap = 4
    target_data = hr.numpy()[::gap]
    trilinear_data = lr.numpy()[::gap]
    sit_data = sit.numpy()[::gap]
    swinir_data = swinir.numpy()[::gap]
    return (target_data, trilinear_data, sit_data, swinir_data,
            hr.numpy(), lr.numpy(), sit.numpy(), swinir.numpy())


(target_data, trilinear_data, sit_data, swinir_data,
 hr_full, lr_full, sit_full, swinir_full) = get_image()

imgs = {
    'Interpolation': trilinear_data,
    '3DRCAN': swinir_data,
    'Ours': sit_data,
    'Target': target_data,
}
vols = {
    'Interpolation': lr_full,
    '3DRCAN': swinir_full,
    'Ours': sit_full,
    'Target': hr_full,
}

rows = ['Interpolation', '3DRCAN', 'Ours', 'Target']
cols_labels = [r'$z_0$', r'$z_0 + 4\mu m$', r'$z_0 + 8\mu m$', r'$z_0 + 12\mu m$', r'$z_0 + 16\mu m$']

roi_x, roi_y = 120, 80
roi_w, roi_h = 90, 90
cut_y = 170
line_color = 'red'
highlight_slice = 4
imshow_args = {'cmap': 'gray', 'vmin': 0, 'vmax': 1, 'interpolation': 'nearest'}

# Main Figure
fig = plt.figure(figsize=(180 * mm, 120 * mm))
gs_master = fig.add_gridspec(1, 2, width_ratios=[3.2, 0.8], wspace=0.1)

# Left: 4x5 grid
gs_left = gs_master[0, 0].subgridspec(4, 5, hspace=0.05, wspace=0.05)

for r, method in enumerate(rows):
    for c in range(5):
        ax = fig.add_subplot(gs_left[r, c])
        if c < len(imgs[method]):
            ax.imshow(imgs[method][c], **imshow_args)
        ax.axis('off')

        if r == 0:
            ax.set_title(cols_labels[c], fontsize=7, pad=3)
        if c == 0:
            ax.text(-0.1, 0.5, method, transform=ax.transAxes,
                    rotation=90, va='center', ha='right', fontsize=7, fontweight='bold')

        # Scale bar on Target row
        if method == 'Target' and c == 0:
            um_per_px_xy = 301.176 / 512.0
            scalebar_um = 50
            scalebar_px = scalebar_um / um_per_px_xy
            x_start, y_start = 10, 256 - 15
            rect = patches.Rectangle((x_start, y_start), scalebar_px, 4,
                                      linewidth=0, facecolor='white', zorder=10)
            ax.add_patch(rect)
            ax.text(x_start + scalebar_px / 2, y_start - 3, rf'{scalebar_um} $\mu$m',
                    color='white', fontsize=7, ha='center', va='bottom', fontweight='bold', zorder=10)

        ax.axhline(y=cut_y, color=line_color, linestyle='--', linewidth=1, alpha=0.8)

# Right: X-Z slices
gs_right = gs_master[0, 1].subgridspec(4, 1, hspace=0.1)
methods_xz = ['Interpolation', '3DRCAN', 'Ours', 'Target']
z_stretch = 4.0

for i, method in enumerate(methods_xz):
    ax_xz = fig.add_subplot(gs_right[i])
    vol = vols[method]
    img_xz = vol[:, cut_y, :]
    aspect_xz = (1.0 / (301.176 / 512.0)) * z_stretch
    ax_xz.imshow(img_xz, cmap='gray', aspect=aspect_xz, interpolation='nearest', vmin=0, vmax=1)

    if i == 0:
        ax_xz.set_title('X-Z Slice', fontsize=7)
    if i == 1:
        ax_xz.set_ylabel(f'z-axis ({int(z_stretch)}x)', fontsize=6, labelpad=1)
        ax_xz.yaxis.set_label_coords(-0.1, -0.1)
    if i == len(methods_xz) - 1:
        ax_xz.set_xlabel('x-axis', fontsize=6, labelpad=1)

    ax_xz.set_xticks([])
    ax_xz.set_yticks([])
    for spine in ax_xz.spines.values():
        spine.set_edgecolor(line_color)
        spine.set_linewidth(1.5)

plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
plt.savefig('../outputs/Figure_2_Panel_c.pdf', dpi=600, transparent=True)
plt.close(fig)

# Supplementary Figure: zoom-in patches
sup_methods = ['Interpolation', '3DRCAN', 'Ours', 'Target']
fig_sup = plt.figure(figsize=(130 * mm, 70 * mm))
gs_sup = fig_sup.add_gridspec(2, 4, hspace=0.15, wspace=0.05)

ax_full = []
ax_zoom = []

for col, method in enumerate(sup_methods):
    # Row 1: Full patch
    ax_f = fig_sup.add_subplot(gs_sup[0, col])
    img_full = imgs[method][highlight_slice]
    ax_f.imshow(img_full, **imshow_args)
    ax_f.set_title(method, fontsize=7, fontweight='bold', pad=3)
    ax_f.axis('off')

    rect = patches.Rectangle((roi_x, roi_y), roi_w, roi_h,
                              linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--')
    ax_f.add_patch(rect)
    ax_full.append(ax_f)

    if method == 'Interpolation':
        um_per_px_xy = 301.176 / 512.0
        scalebar_um_f = 50
        scalebar_px_f = scalebar_um_f / um_per_px_xy
        x_start_f, y_start_f = 10, 256 - 15
        rect_f = patches.Rectangle((x_start_f, y_start_f), scalebar_px_f, 4,
                                   linewidth=0, facecolor='white', zorder=50)
        ax_f.add_patch(rect_f)
        ax_f.text(x_start_f + scalebar_px_f / 2, y_start_f - 3, rf'{scalebar_um_f} $\mu$m',
                  color='white', fontsize=7, ha='center', va='bottom', fontweight='bold', zorder=50)

    # Row 2: Zoom-in patch
    ax_z = fig_sup.add_subplot(gs_sup[1, col])
    zoom_img = img_full[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    ax_z.imshow(zoom_img, **imshow_args)
    ax_z.set_xticks([])
    ax_z.set_yticks([])
    for spine in ax_z.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('red')
        spine.set_linewidth(1.5)

    if method == 'Interpolation':
        scalebar_um_z = 10
        scalebar_px_z = scalebar_um_z / um_per_px_xy
        x_start_z, y_start_z = 5, roi_h - 10
        rect_z = patches.Rectangle((x_start_z, y_start_z), scalebar_px_z, 2,
                                   linewidth=0, facecolor='white', zorder=50)
        ax_z.add_patch(rect_z)
        ax_z.text(x_start_z + scalebar_px_z / 2, y_start_z - 2, rf'{scalebar_um_z} $\mu$m',
                  color='white', fontsize=7, ha='center', va='bottom', fontweight='bold', zorder=50)

    ax_zoom.append(ax_z)

    # Connection lines from ROI to zoom
    con_args = dict(coordsA="data", coordsB="data",
                    color="red", linestyle="--", linewidth=1.0, alpha=0.9,
                    arrowstyle="-", clip_on=False, zorder=5)
    dest_h, dest_w = zoom_img.shape
    con1 = ConnectionPatch(xyA=(roi_x, roi_y + roi_h), xyB=(0, 0),
                           axesA=ax_f, axesB=ax_z, **con_args)
    ax_f.add_artist(con1)
    con2 = ConnectionPatch(xyA=(roi_x + roi_w, roi_y + roi_h), xyB=(dest_w - 0.5, 0),
                           axesA=ax_f, axesB=ax_z, **con_args)
    ax_f.add_artist(con2)

plt.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.05)
plt.savefig('../outputs/Sup_Fig_2.pdf', dpi=600, transparent=True)
plt.close(fig_sup)
