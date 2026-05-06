# Figure Reproduction Scripts

Plotting scripts used to generate figures for the paper.

## Setup

Each script reads from one or more HDF5 files containing model predictions
and ground truth data. Inside every script, replace the `<YOUR_DATA_PATH>`
placeholder with the absolute path to your local results / metrics HDF5 file.

Expected HDF5 contents:

- Results files: `hr` (ground truth), `lr` (input), and one dataset per
  method holding the model prediction volume — e.g. `microdiffuse3d`
  (ours), `microdiffuse3d_VAEdecoder` (VAE-decoder ablation),
  `RCAN_output`, `SwinIR_output`, `RLN_output`, `CARE_output`,
  `SRCNN_output`.
- Metrics files: per-sample arrays keyed as `{method}_{metric}` for
  `metric ∈ {ssim, psnr, msssim, lpips}` and the methods listed in
  `utils.build_long_dataframe`.

Outputs are written to `../outputs/`.

## Scripts

| Script | Main figure(s) | Supplementary figure(s) |
|--------|----------------|-------------------------|
| `f1_cube_visual.py` | Figure 1 — 3D MAE-style masked cube visualization | — |
| `f2pab.py` | Figure 2 panels a,b — Quantitative metric box plots (3DSR + BioTISR) | — |
| `f2pc.py` | Figure 2 panel c — Qualitative Z-slice + X-Z views | Sup Fig 2 — Zoom-in patches |
| `f3pab.py` | Figure 3 panels a,b — Denoising example + metric box plots | — |
| `f3pc.py` | Figure 3 panel c — Denoising qualitative Z-slice + X-Z views | Sup Fig 3 — Zoom-in patches |
| `f4pa.py` | Figure 4 panel a — Cellpose segmentation (PQ + Dice violins) | Sup Fig 4 — F1 violins |
| `f4pcd.py` | Figure 4 panels c,d — 1D diagonal profile + CCC regression/violin | Sup Fig 5 — PCC + 1D-SSIM violins |
| `supf6.py` | — | Sup Fig 6 — Decoder ablation violin plots |
| `supf7.py` | — | Sup Fig 7 — Frequency analysis & perception–distortion tradeoff |
| `supf8.py` | — | Sup Fig 8 — Laplacian variance comparison |

`nature_style.py` provides shared Nature sub-journal styling, and
`utils.py` provides the shared `build_long_dataframe` helper that loads
per-sample metrics into a long-format DataFrame.

## Dependencies

- `matplotlib`
- `seaborn`
- `numpy`
- `pandas`
- `h5py`
- `scipy`
- `torch`
- `scikit-image`
- `cellpose` (only for `f4pa.py`)
- `opencv-python` (only for `f4pa.py`)
- `imageio` (only for `utils.create_comparison_gif`)
- `tqdm`
