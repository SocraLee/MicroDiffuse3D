"""
cube_visual.py
==============
3D MAE-style masked cube visualization with real brain data.

Panels:  HR (with grid)  →  Masked HR (75%)  →  LR
- Actual pixel data on cube faces
- Physically correct coordinates: XY 1µm=1.7px,  Z 1µm=1px
- z_stretch parameter to visually fatten the depth axis
- Masked panel: each surviving patch rendered as textured 3D box

Usage:
    python cube_visual.py --idx 20
    python cube_visual.py --idx 20 --z_stretch 3
    python cube_visual.py --hires --save out.pdf
"""

import argparse, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nature_style

# ════════════════════ Physical constants ════════════════════
XY_PIXEL_SIZE = 1.0 / 1.7   # µm per pixel in XY
Z_PIXEL_SIZE  = 1.0          # µm per pixel in Z

# ════════════════════ Data ════════════════════

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

def load_data(idx=1281, data_path=None, use_cache=False):
    """Return (hr, lr, pred).  pred may be None if key not found."""
    import h5py
    if data_path is None:
        # 3DSR HDF5 with `hr`, `lr`, and `microdiffuse3d_VAEdecoder` model prediction dataset.
        data_path = '<YOUR_DATA_PATH>'

    with h5py.File(data_path, 'r') as f:
        hr = np.array(f['hr'][idx]).squeeze()
        lr = np.array(f['lr'][idx]).squeeze()
        pred = None
        if 'microdiffuse3d_VAEdecoder' in f:
            pred = np.array(f['microdiffuse3d_VAEdecoder'][idx]).squeeze()
    return hr, lr, pred

def generate_mask(nd, nh, nw, ratio=0.75, seed=42):
    rng = np.random.RandomState(seed)
    n = nd * nh * nw
    f = np.ones(n, dtype=bool)
    f[rng.choice(n, int(round(n * ratio)), replace=False)] = False
    return f.reshape(nd, nh, nw)

# ════════════════════ Coordinate helpers ════════════════════
#
# Coordinate mapping  (all in µm):
#   plot-x  ←  depth (D)  × Z_PIXEL_SIZE × z_stretch     (brain z → RIGHT)
#   plot-y  ←  width (W)  × XY_PIXEL_SIZE                 (→ BACK)
#   plot-z  ←  height (H) × XY_PIXEL_SIZE                 (→ UP)
#
# sx, sy, sz  are the per-pixel scale factors for each axis.

def _scales(z_stretch):
    """Return (sx, sy, sz) per-pixel → µm scale factors."""
    return (Z_PIXEL_SIZE * z_stretch, XY_PIXEL_SIZE, XY_PIXEL_SIZE)

# ════════════════════ Outer-face rendering (full cubes) ════════════════════

def _sub(data, mx):
    h, w = data.shape
    return data[::max(1,h//mx), ::max(1,w//mx)]

def draw_face(ax, face, vol, nd, nh, nw, pd, ph, pw, sc, mx=256):
    """Render one outer face of a FULL cube with pixel data.
    sc = (sx, sy, sz) per-pixel scale factors."""
    sx, sy, sz = sc
    cm = plt.get_cmap('gray')
    nm = mcolors.Normalize(vmin=0, vmax=1)
    uD, uH, uW = nd*pd, nh*ph, nw*pw

    if face == 'left':          # x = 0,  y=W, z=H
        d = _sub(vol[0, :uH, :uW], mx)
        rH, rW = d.shape
        y = np.linspace(0, uW*sy, rW+1)
        z = np.linspace(0, uH*sz, rH+1)
        Y, Z = np.meshgrid(y, z, indexing='ij')
        ax.plot_surface(np.zeros_like(Y), Y, Z,
                        facecolors=cm(nm(d[::-1].T)), shade=False,
                        rstride=1, cstride=1, antialiased=False)
    elif face == 'front':       # y = 0,  x=D, z=H
        d = _sub(vol[:uD, :uH, 0], mx)
        rD, rH = d.shape
        x = np.linspace(0, uD*sx, rD+1)
        z = np.linspace(0, uH*sz, rH+1)
        X, Z = np.meshgrid(x, z, indexing='ij')
        ax.plot_surface(X, np.zeros_like(X), Z,
                        facecolors=cm(nm(d[:,::-1])), shade=False,
                        rstride=1, cstride=1, antialiased=False)
    elif face == 'top':         # z = uH*sz,  x=D, y=W
        d = _sub(vol[:uD, 0, :uW], mx)
        rD, rW = d.shape
        x = np.linspace(0, uD*sx, rD+1)
        y = np.linspace(0, uW*sy, rW+1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        ax.plot_surface(X, Y, np.full_like(X, uH*sz),
                        facecolors=cm(nm(d)), shade=False,
                        rstride=1, cstride=1, antialiased=False)

# ════════════════════ Masked-patch rendering ════════════════════

def draw_masked_patches(ax, vol, vis, nd, nh, nw, pd, ph, pw, sc):
    sx, sy, sz = sc
    cm = plt.get_cmap('gray')
    nm = mcolors.Normalize(vmin=0, vmax=1)

    for id_ in range(nd):
        for ih in range(nh):
            for iw in range(nw):
                if not vis[id_, ih, iw]:
                    continue
                # Plot-space extents (µm)
                x0 = id_*pd*sx;      x1 = x0 + pd*sx
                y0 = iw*pw*sy;       y1 = y0 + pw*sy
                z0 = (nh-1-ih)*ph*sz; z1 = z0 + ph*sz

                ds, de = id_*pd, (id_+1)*pd
                hs, he = ih*ph, (ih+1)*ph
                ws, we = iw*pw, (iw+1)*pw

                faces = []
                if id_==0    or not vis[id_-1,ih,iw]: faces.append('xn')
                if id_==nd-1 or not vis[id_+1,ih,iw]: faces.append('xp')
                if iw==0     or not vis[id_,ih,iw-1]: faces.append('yn')
                if iw==nw-1  or not vis[id_,ih,iw+1]: faces.append('yp')
                if ih==nh-1  or not vis[id_,ih+1,iw]: faces.append('zn')
                if ih==0     or not vis[id_,ih-1,iw]: faces.append('zp')

                for fn in faces:
                    _draw_patch_face(ax, fn, vol,
                                     ds,de,hs,he,ws,we,
                                     x0,x1,y0,y1,z0,z1, cm, nm)


def _draw_patch_face(ax, fn, vol, ds,de,hs,he,ws,we, x0,x1,y0,y1,z0,z1, cm, nm):
    kw = dict(shade=False, rstride=1, cstride=1, antialiased=False)
    if fn in ('xn', 'xp'):
        d_idx = ds if fn=='xn' else de-1
        data = vol[d_idx, hs:he, ws:we]
        rH, rW = data.shape
        y = np.linspace(y0, y1, rW+1)
        z = np.linspace(z0, z1, rH+1)
        Y, Z = np.meshgrid(y, z, indexing='ij')
        X = np.full_like(Y, x0 if fn=='xn' else x1)
        ax.plot_surface(X, Y, Z, facecolors=cm(nm(data[::-1].T)), **kw)
    elif fn in ('yn', 'yp'):
        w_idx = ws if fn=='yn' else we-1
        data = vol[ds:de, hs:he, w_idx]
        rD, rH = data.shape
        x = np.linspace(x0, x1, rD+1)
        z = np.linspace(z0, z1, rH+1)
        X, Z = np.meshgrid(x, z, indexing='ij')
        Y = np.full_like(X, y0 if fn=='yn' else y1)
        ax.plot_surface(X, Y, Z, facecolors=cm(nm(data[:,::-1])), **kw)
    elif fn in ('zn', 'zp'):
        h_idx = he-1 if fn=='zn' else hs
        data = vol[ds:de, h_idx, ws:we]
        rD, rW = data.shape
        x = np.linspace(x0, x1, rD+1)
        y = np.linspace(y0, y1, rW+1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = np.full_like(X, z0 if fn=='zn' else z1)
        ax.plot_surface(X, Y, Z, facecolors=cm(nm(data)), **kw)

# ════════════════════ Grid / Wireframe ════════════════════

def draw_outer_grid(ax, nd, nh, nw, pd, ph, pw, sc,
                    color='#333', lw=0.5, alpha=0.6):
    sx, sy, sz = sc
    tx, ty, tz = nd*pd*sx, nw*pw*sy, nh*ph*sz
    for fx in [0, tx]:
        for i in range(nh+1):
            ax.plot3D([fx,fx],[0,ty],[i*ph*sz,i*ph*sz], color=color, lw=lw, alpha=alpha)
        for i in range(nw+1):
            ax.plot3D([fx,fx],[i*pw*sy,i*pw*sy],[0,tz], color=color, lw=lw, alpha=alpha)
    for fy in [0, ty]:
        for i in range(nh+1):
            ax.plot3D([0,tx],[fy,fy],[i*ph*sz,i*ph*sz], color=color, lw=lw, alpha=alpha)
        for i in range(nd+1):
            ax.plot3D([i*pd*sx,i*pd*sx],[fy,fy],[0,tz], color=color, lw=lw, alpha=alpha)
    for fz in [0, tz]:
        for i in range(nd+1):
            ax.plot3D([i*pd*sx,i*pd*sx],[0,ty],[fz,fz], color=color, lw=lw, alpha=alpha)
        for i in range(nw+1):
            ax.plot3D([0,tx],[i*pw*sy,i*pw*sy],[fz,fz], color=color, lw=lw, alpha=alpha)

def draw_full_wireframe(ax, nd, nh, nw, pd, ph, pw, sc,
                        color='#888', lw=0.3, alpha=0.25):
    sx, sy, sz = sc
    tx, ty, tz = nd*pd*sx, nw*pw*sy, nh*ph*sz
    for ih in range(nh+1):
        for iw in range(nw+1):
            ax.plot3D([0,tx],[iw*pw*sy,iw*pw*sy],[ih*ph*sz,ih*ph*sz], c=color, lw=lw, alpha=alpha)
    for ih in range(nh+1):
        for id_ in range(nd+1):
            ax.plot3D([id_*pd*sx,id_*pd*sx],[0,ty],[ih*ph*sz,ih*ph*sz], c=color, lw=lw, alpha=alpha)
    for id_ in range(nd+1):
        for iw in range(nw+1):
            ax.plot3D([id_*pd*sx,id_*pd*sx],[iw*pw*sy,iw*pw*sy],[0,tz], c=color, lw=lw, alpha=alpha)

def style_ax(ax, title, nd, nh, nw, pd, ph, pw, sc):
    sx, sy, sz = sc
    tx, ty, tz = nd*pd*sx, nw*pw*sy, nh*ph*sz
    m = max(tx,ty,tz)*0.05
    ax.set_xlim(-m,tx+m); ax.set_ylim(-m,ty+m); ax.set_zlim(-m,tz+m)
    # Force the 3D box to respect actual physical proportions
    ax.set_box_aspect([tx, ty, tz])
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, fontweight='bold', pad=-5, y=-0.02)

# ════════════════════ Main ════════════════════

def make_figure(idx=1281, data_path=None,
                patch_d=5, patch_hw=64,
                crop_hw=None,
                mask_ratio=0.75, seed=42,
                z_stretch=3.0, elev=25, azim=220,
                grid_lw=0.5, grid_alpha=0.6,
                max_face_res=128,
                hires=False, save_path=None):
    """
    crop_hw   : int or None – crop XY to this size (center crop). None = full.
    z_stretch = 1.0 → physically correct  (cube ≈ 20 × 150 × 150 µm, very thin)
    z_stretch > 1   → fatten the depth for visibility
    """
    mm = nature_style.apply_nature_style()
    hr, lr, pred = load_data(idx, data_path)

    # Center-crop XY if requested
    if crop_hw is not None:
        _, H_full, W_full = hr.shape
        h0 = (H_full - crop_hw) // 2
        w0 = (W_full - crop_hw) // 2
        hr = hr[:, h0:h0+crop_hw, w0:w0+crop_hw]
        lr = lr[:, h0:h0+crop_hw, w0:w0+crop_hw]
        if pred is not None:
            pred = pred[:, h0:h0+crop_hw, w0:w0+crop_hw]

    D_hr, H, W = hr.shape;  D_lr = lr.shape[0]
    has_pred = pred is not None
    n_cols = 4 if has_pred else 3

    sc_hr = _scales(z_stretch)

    nd_hr = D_hr // patch_d
    nh = H // patch_hw;  nw = W // patch_hw

    # LR: same visual z-length as HR
    if D_lr % nd_hr == 0:
        pd_lr, nd_lr = D_lr // nd_hr, nd_hr
    else:
        pd_lr, nd_lr = 1, D_lr
    target_z_um = nd_hr * patch_d * sc_hr[0]
    sx_lr = target_z_um / (nd_lr * pd_lr)
    sc_lr = (sx_lr, XY_PIXEL_SIZE, XY_PIXEL_SIZE)

    masked_vis = generate_mask(nd_hr, nh, nw, mask_ratio, seed)

    phys_z = D_hr * Z_PIXEL_SIZE
    phys_xy = H * XY_PIXEL_SIZE


    dpi = 300 if hires else 200
    fig = plt.figure(figsize=(210*mm, 120*mm), dpi=dpi)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, n_cols, height_ratios=[2, 1], hspace=0.05, wspace=0.05)

    # ---- Row 1: 3D cubes ----
    col = 0

    # Panel: HR
    ax1 = fig.add_subplot(gs[0, col], projection='3d')
    for face in ['left','front','top']:
        draw_face(ax1, face, hr, nd_hr, nh, nw, patch_d, patch_hw, patch_hw,
                  sc_hr, mx=max_face_res)
    ax1.view_init(elev=elev, azim=azim)
    style_ax(ax1, 'HR', nd_hr, nh, nw, patch_d, patch_hw, patch_hw, sc_hr)
    col += 1

    # Panel: Masked HR
    ax2 = fig.add_subplot(gs[0, col], projection='3d')
    draw_masked_patches(ax2, hr, masked_vis, nd_hr, nh, nw,
                        patch_d, patch_hw, patch_hw, sc_hr)
    draw_full_wireframe(ax2, nd_hr, nh, nw, patch_d, patch_hw, patch_hw,
                        sc_hr, lw=0.8, alpha=0.5)
    draw_outer_grid(ax2, nd_hr, nh, nw, patch_d, patch_hw, patch_hw, sc_hr,
                    lw=1, alpha=0.7)
    ax2.view_init(elev=elev, azim=azim)
    style_ax(ax2, 'Masked HR (%.0f%%)'%(mask_ratio*100),
             nd_hr, nh, nw, patch_d, patch_hw, patch_hw, sc_hr)
    col += 1

    # Panel: Prediction (if available)
    if has_pred:
        D_pred = pred.shape[0]
        nd_pred = D_pred // patch_d
        if D_pred == D_hr:
            sc_pred = sc_hr
            pd_pred, nd_pred = patch_d, nd_hr
        else:
            pd_pred = 1; nd_pred = D_pred
            sc_pred = (target_z_um / (nd_pred * pd_pred), XY_PIXEL_SIZE, XY_PIXEL_SIZE)
        ax_pred = fig.add_subplot(gs[0, col], projection='3d')
        nh_p = pred.shape[1] // patch_hw; nw_p = pred.shape[2] // patch_hw
        for face in ['left','front','top']:
            draw_face(ax_pred, face, pred, nd_pred, nh_p, nw_p, pd_pred,
                      patch_hw, patch_hw, sc_pred, mx=max_face_res)
        ax_pred.view_init(elev=elev, azim=azim)
        style_ax(ax_pred, 'Prediction', nd_pred, nh_p, nw_p, pd_pred, patch_hw, patch_hw, sc_pred)
        col += 1

    # Panel: LR
    ax3 = fig.add_subplot(gs[0, col], projection='3d')
    nh_lr = lr.shape[1] // patch_hw;  nw_lr = lr.shape[2] // patch_hw
    for face in ['left','front','top']:
        draw_face(ax3, face, lr, nd_lr, nh_lr, nw_lr, pd_lr, patch_hw,
                  patch_hw, sc_lr, mx=max_face_res)
    ax3.view_init(elev=elev, azim=azim)
    style_ax(ax3, 'LR', nd_lr, nh_lr, nw_lr, pd_lr, patch_hw, patch_hw, sc_lr)

    # ---- Row 2: center z-slice under HR, Prediction, LR ----
    center_z = D_hr // 2
    imkw = dict(cmap='gray', vmin=0, vmax=1, interpolation='nearest', aspect='equal')

    # Slice positions: HR at col 0, skip Masked col, Pred, LR
    ax_hr_slice = fig.add_subplot(gs[1, 0])
    ax_hr_slice.imshow(hr[center_z], **imkw)
    ax_hr_slice.set_title(f'HR  (z = {center_z})', fontsize=7)
    ax_hr_slice.axis('off')

    # Masked column empty
    ax_empty = fig.add_subplot(gs[1, 1])
    ax_empty.axis('off')

    if has_pred:
        center_z_pred = pred.shape[0] // 2
        ax_pred_slice = fig.add_subplot(gs[1, 2])
        ax_pred_slice.imshow(pred[center_z_pred], **imkw)
        ax_pred_slice.set_title(f'Prediction  (z = {center_z_pred})', fontsize=7)
        ax_pred_slice.axis('off')

    lr_col = n_cols - 1
    center_z_lr = D_lr // 2
    ax_lr_slice = fig.add_subplot(gs[1, lr_col])
    ax_lr_slice.imshow(lr[center_z_lr], **imkw)
    ax_lr_slice.set_title(f'LR  (z = {center_z_lr})', fontsize=7)
    ax_lr_slice.axis('off')


    plt.subplots_adjust(left=0.01,right=0.99,bottom=0.02,top=0.92,wspace=0.05)

    if save_path:
        plt.savefig(save_path, dpi=dpi, transparent=True)

    plt.show()
    return fig

# ════════════════════ CLI ════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--idx',          type=int,   default=1281)
    p.add_argument('--data_path',    type=str,   default = None)
    p.add_argument('--mask_ratio',   type=float, default=0.75)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--patch_d',      type=int,   default=5)
    p.add_argument('--patch_hw',     type=int,   default=64)
    p.add_argument('--crop_hw',      type=int,   default=None,
                   help='Center-crop XY to this size (e.g. 64, 128). None=full')
    p.add_argument('--z_stretch',    type=float, default=3.0,
                   help='1.0 = physically correct; >1 fattens depth axis')
    p.add_argument('--elev',         type=float, default=25)
    p.add_argument('--azim',         type=float, default=220)
    p.add_argument('--max_face_res', type=int,   default=128)
    p.add_argument('--hires',        action='store_true')
    p.add_argument('--save',         type=str,   default='../outputs/cube_visual.png')
    return p.parse_args()

if __name__ == '__main__':
    a = parse_args()
    make_figure(idx=a.idx,data_path = a.data_path, mask_ratio=a.mask_ratio, seed=a.seed,
                patch_d=a.patch_d, patch_hw=a.patch_hw, crop_hw=a.crop_hw,
                z_stretch=a.z_stretch,
                elev=a.elev, azim=a.azim, max_face_res=a.max_face_res,
                hires=a.hires, save_path=a.save)
