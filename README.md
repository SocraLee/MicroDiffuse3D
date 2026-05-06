# MicroDiffuse3D

**A Foundation Model for Volumetric Microscopy Image Restoration**

This repository contains the source code accompanying the following manuscript in preparation:

> **MicroDiffuse3D: A Foundation Model for 3D Microscopy Imaging Restoration**

MicroDiffuse3D is a conditional diffusion model for volumetric (3D) microscopy image restoration. It addresses multiple restoration tasks within a single unified framework:

- **3D Super-Resolution** — Recovers high-resolution lateral and axial information from sparsely-sampled Z-stacks (e.g., 4× lateral & 4× axial)
- **3D Denoising** — Removes noise from low signal-to-noise ratio (SNR) volumetric acquisitions
- **Joint Degradation Restoration** — Jointly addresses coupled degradation of image quality and resolution

## Architecture

MicroDiffuse3D combines:
1. **SiT-3D Backbone** — A 3D Diffusion Transformer with Anisotropic Lateral-Axial Attention for efficient processing of volumetric data
2. **3D UNet Conditioning Module** — Encodes low-resolution input volumes into conditioning signals
3. **Post-Diffusion Decoders** — Specialized decoder variants that map VAE latents back to pixel space for high-fidelity detail refinement
4. **REPA Alignment (optional)** — REPresentation Alignment loss using DINOv2 features. We omitted this technique from our main results due to the high computational cost of preprocessing massive pretraining datasets. However, our initial experiments showed that it accelerates convergence and reduces data requirements. We have retained this feature in the codebase to facilitate adaptation to specific downstream tasks, particularly in domains where data is limited

## Installation
### Step 1: Clone the repository
```bash
git clone https://github.com/SocraLee/MicroDiffuse3D.git
cd MicroDiffuse3D
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
conda create -n microdiffuse3d python=3.10 -y
conda activate microdiffuse3d
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Multi-GPU Training

```bash
accelerate config
```

---

## Getting Started

This section provides a guide on how to prepare your data, configure the model, and run training and inference pipelines.

### Data Organization

Training and evaluation data should be organized as HDF5 files containing the following structure:

| Key            | Shape                  | Description                                                     |
| -------------- | ---------------------- | --------------------------------------------------------------- |
| `hr_cube`      | `(N, D, H, W)`         | High-quality ground truth volumes                               |
| `lr_cube`      | `(N, D_lr, H, W)`      | Low-quality input volumes                                       |
| `vae_hr_cube`  | `(N, D, 8, 32, 32)`    | Precomputed VAE latents (mean + std)                            |
| `dino_lr_cube` | `(N, D_lr, 256, 1024)` | Precomputed DINOv2 embeddings of LR slices                      |
| `dino_hr_cube` | `(N, D, 256, 1024)`    | Precomputed DINOv2 embeddings of HR slices (optional, for REPA) |

To preprocess your own raw volumes into this format, use the feature extraction script:

```bash
python data_processing/prepare_features.py \
    --data_path <path_to_your_h5_file>
```

### Configuration

All tasks are defined by configuration JSON files in the `configs/` directory:

1. **Model Backbone:** Configure SiT-3D hyperparameters in `configs/backbone_config.json`.
2. **Conditioning:** Configure the LR encoder in `configs/encoder_config.json`.
3. **Data Paths:** In task-specific configs (e.g., `configs/3dsr4z_config.json`), update `train_data_dir` and `dev_data_dir` to point to your preprocessed HDF5 directories.

### Training the Model

**Single-task training (e.g., 3D Super-Resolution 4×):**

```bash
accelerate launch \
    --gpu_ids 0,1 \
    --num_processes 2 \
    microdiffuse3d/train.py \
    --output-dir ./outputs \
    --exp-name 3dsr4z_experiment \
    --backbone_args_json ./configs/backbone_config.json \
    --task_configs_json ./configs/3dsr4z_config.json \
    --conditioning_args_json ./configs/encoder_config.json
```

**Multi-task joint training:**

Use `configs/task_config.json` to define multiple tasks with respective sampling weights.

*Note: For Representation Alignment (REPA), append `--proj-coeff 0.5 --z-types dinov2 --z-weights 1.0` to the command.*

### Finetuning the Post-Diffusion Decoder

To further refine pixel-space artifacts, you can train and apply an adapted post-diffusion decoder.

```bash
python microdiffuse3d/train_decoder.py \
    --task_configs_json configs/3dsr4z_config.json \
    --exp-name 3dsr4z_adapted \
    --save-dir ./outputs/decoder
```

### Generation / Inference

Generate high-resolution predictions using a trained checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python microdiffuse3d/generate.py \
    --task_configs_json ./configs/3dsr4z_config.json \
    --backbone_args_json ./configs/backbone_config.json \
    --conditioning_args_json ./configs/encoder_config.json \
    --ckpt_path ./outputs/<exp_name>/checkpoints/acc_step_<step>/model.pt \
    --exp-name eval_results
```

Evaluate with the finetuned decoder:

```bash
CUDA_VISIBLE_DEVICES=0 python microdiffuse3d/generate.py \
    --task_configs_json ./configs/3dsr4z_config.json \
    --backbone_args_json ./configs/backbone_config.json \
    --conditioning_args_json ./configs/encoder_config.json \
    --ckpt_path ./outputs/<exp_name>/checkpoints/acc_step_<step>/model.pt \
    --exp-name eval_adapted \
    --decoder-type adapted \
    --decoder-path ./outputs/decoder/adapted/3d_sr/best_decoder.pth
```

Predicted volumes will be saved directly into the input HDF5 file under the key `microdiffuse3d_output`. Console output will report PSNR and SSIM.

---

## Reproducing Paper Results

To directly reproduce the results reported in the manuscript, download the pre-trained checkpoints and processed datasets.

### Pre-trained Checkpoint
The unified multi-task foundation model checkpoint can be downloaded here:
[**Download Pre-trained Checkpoint**](https://zenodo.org/records/19959440)

### Processed Datasets

Processed HDF5 datasets containing VAE and DINO embeddings are hosted on Zenodo. Download the respective splits below:

|     Split      | SRS 3D Super-Resolution                                        | SRS 3D Denoise                                                 | BioTISR                                               |
| :------------: | :------------------------------------------------------------- | :------------------------------------------------------------- | :------------------------------------------------------------- |
|   **Train**    | [Link (19929571)](https://zenodo.org/records/19929571)         | [Link (19929547)](https://zenodo.org/records/19929547)         | [Link (19938235)](https://zenodo.org/records/19938235)         |
| **Validation** | [Unified Link (19941581)](https://zenodo.org/records/19941581) | [Unified Link (19941581)](https://zenodo.org/records/19941581) | [Unified Link (19941581)](https://zenodo.org/records/19941581) |
|    **Test**    | [Unified Link (19946789)](https://zenodo.org/records/19946789) | [Unified Link (19946789)](https://zenodo.org/records/19946789) | [Unified Link (19946789)](https://zenodo.org/records/19946789) |

### Reproducing Figures

Figure reproduction scripts are provided in the `figures/` directory. Edit `figures/config.py` to set the correct paths to your downloaded test files and results, then run individual scripts:

```bash
python figures/f2pab.py   # Figure 2 panels a,b
python figures/f3pab.py   # Figure 3 panels a,b
```

---

## Project Structure

```
MicroDiffuse3D/
├── microdiffuse3d/               # Main Python package
│   ├── train.py                 # Multi-task distributed training
│   ├── generate.py              # Inference and evaluation
│   ├── train_decoder.py         # Post-diffusion decoder training
│   ├── loss.py                  # SILoss (denoising + REPA)
│   ├── samplers.py              # Euler & Euler-Maruyama SDE/ODE samplers
│   ├── model/
│   │   ├── model.py             # JointModel top-level wrapper
│   │   ├── sit.py               # Timestep embedder, condition fuser, projectors
│   │   ├── sit_backbone.py      # SiT-3D backbone with factorized attention
│   │   ├── attention.py         # Anisotropic lateral-axial attention
│   │   ├── unet3d.py            # LR conditioning encoder-decoder
│   │   ├── decoder.py           # Adapted & Fused post-diffusion decoders
│   │   ├── utils.py             # Positional embeddings
│   │   └── rope.py              # Rotary position embeddings
│   └── data/
│       ├── dataset.py           # HDF5-based dataset classes
│       ├── dataloader.py        # Multi-task weighted data loading
│       └── data_utils.py        # PSF simulation and degradation utilities
├── configs/                     # Model and task configuration files
├── scripts/                     # Example shell scripts for training and evaluation
├── data_processing/             # Data preprocessing utilities
├── figures/                     # Figure reproduction scripts
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md                    # This file
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{li2026microdiffuse3d,
  title   = {MicroDiffuse3D: A Foundation Model for 3D Microscopy Imaging Restoration},
  author  = {Yongkang Li et al.},
  <!-- journal = {Nature Methods}, -->
  year    = {2026}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

This work builds upon the following open-source projects:

- [Scalable Interpolant Transformers (SiT)](https://github.com/willisma/SiT)
- [Stable Diffusion VAE](https://github.com/CompVis/stable-diffusion)
- [DINOv2](https://github.com/facebookresearch/dinov2)

---

## Contact

For questions or issues regarding this code, please open a [GitHub Issue](https://github.com/SocraLee/MicroDiffuse3D/issues) or contact the corresponding author.
