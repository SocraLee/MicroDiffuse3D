"""
MicroDiffuse3D Inference / Evaluation Script.

Loads a trained model checkpoint and runs evaluation on test data,
computing PSNR and SSIM metrics. Supports optional post-diffusion
decoder refinement (Adapted decoder, Finetuned VAE decoder).

Usage:
    CUDA_VISIBLE_DEVICES=0 python generate.py \
        --task_configs_json ./configs/3dsr4z_config.json \
        --backbone_args_json ./configs/backbone_config.json \
        --conditioning_args_json ./configs/encoder_config.json \
        --ckpt_path /path/to/model.pt \
        --exp-name eval_40k \
        --save-results \
        --batch-size 4 \
        --num-steps 50
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import json
import os
import time
import logging

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from diffusers.models import AutoencoderKL
from torch_ema import ExponentialMovingAverage
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from model.model import JointModel
from model.utils import sample_posterior
from model.decoder import AdaptedDecoder
from data.dataset import H5Dataset
from samplers import euler_maruyama_sampler

logger = logging.getLogger(__name__)

def get_test_path(train_path):
    """Derive test path from training path."""
    if "/train/" in train_path:
        return train_path.replace("/train/", "/test/")
    return train_path.replace("train", "test")

def get_val_path(train_path):
    """Derive validation path from training path."""
    if "/train/" in train_path:
        return train_path.replace("/train/", "/val/")
    return train_path.replace("train", "val")

def generate_and_decode(model, latents_noise, cond_input, latents_scale, latents_bias,
                        args, lr_image=None, decoder=None):
    """Generate latent samples and decode to image space."""
    samples = euler_maruyama_sampler(
        model,
        latents_noise,
        cond_input,
        num_steps=args.num_steps,
        path_type=args.path_type,
        cfg_scale=1.0,
        guidance_low=0.,
        guidance_high=1.,
        heun=False,
    ).to(torch.float32)

    if args.decoder_type == 'adapted' and decoder is not None and lr_image is not None:
        # Use adapted post-diffusion decoder
        B, D, C, H, W = samples.shape
        vae_latent = samples[:, :, :4].permute(0, 2, 1, 3, 4)  # (B, 4, D, H, W)
        output = decoder(lr_image, vae_latent)
        output = output.permute(0, 2, 1, 3, 4)  # (B, D, 1, H_out, W_out)
        B, D, C, H, W = output.shape
        return output.reshape(B * D, C, H, W).clamp(0, 1)
    else:
        # Use VAE decoder (pretrained or finetuned)
        B, D, C, H, W = samples.shape
        samples = samples.reshape(B * D, C, H, W)
        samples = decoder.decode((samples - latents_bias) / latents_scale).sample
        samples = (samples + 1) / 2.
        return samples.clamp(0, 1)


def run_evaluation(args):
    """Main evaluation loop."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load configs
    with open(args.task_configs_json, 'r') as f:
        task_configs = json.load(f)
    with open(args.backbone_args_json, 'r') as f:
        backbone_args = json.load(f)
    with open(args.conditioning_args_json, 'r') as f:
        conditioning_args = json.load(f)

    common_z_types = args.z_types

    latent_size = args.resolution // 8
    backbone_args["input_size"] = latent_size

    # Build model
    model = JointModel(
        backbone_args=backbone_args,
        conditioning_args=conditioning_args,
        common_z_types=common_z_types,
        task_configs=task_configs,
    ).to(device)

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load checkpoint
    state_dict = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logger.info(f"Loaded checkpoint from {args.ckpt_path}")
    # Load VAE
    latents_scale = torch.tensor([0.18215] * 4).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor([0.] * 4).view(1, 4, 1, 1).to(device)

    # Load optional decoder
    decoder = None
    if args.decoder_type == 'adapted':
        decoder = AdaptedDecoder().to(device)
        if args.decoder_path and os.path.exists(args.decoder_path):
            state_dict = torch.load(args.decoder_path, map_location=device, weights_only=True)
            decoder.load_state_dict(state_dict)
            logger.info(f"Loaded adapted decoder from {args.decoder_path}")
        else:
            logger.warning(f"Adapted decoder path not found: {args.decoder_path}. Using random initialization.")
        decoder.eval()
    else:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
        if args.decoder_type == 'finetuned_vae':
            if args.decoder_path and os.path.exists(args.decoder_path):
                state_dict = torch.load(args.decoder_path, map_location=device, weights_only=True)
                vae.decoder.load_state_dict(state_dict)
                logger.info(f"Loaded finetuned VAE decoder from {args.decoder_path}")
            else:
                logger.warning(f"VAE decoder path not found: {args.decoder_path}. Using pretrained VAE decoder.")
        decoder = vae
    # Metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0, dim=(1, 2, 3)).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Run on each task
    for task_id, config in task_configs.items():
        test_path = get_test_path(config['train_data_dir'])
        if not os.path.exists(test_path):
            logger.warning(f"Test data not found for {task_id}: {test_path}")
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating task: {task_id}")
        logger.info(f"Test data: {test_path}")

        test_dataset = H5Dataset(test_path, if_train=False)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )

        all_preds = []
        all_gts = []
        psnr_metric.reset()
        ssim_metric.reset()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"[{task_id}]")):
                hr_image, dino_lr, vae_hr_x, _, lr_image = batch

                hr_image = hr_image.to(device)
                dino_lr = dino_lr.to(device)
                vae_hr_x = vae_hr_x.to(device)
                lr_image = lr_image.to(device)

                vae_hr_x = sample_posterior(vae_hr_x, latents_scale, latents_bias)
                latents_noise = torch.randn_like(vae_hr_x).to(device)

                cond_input = (dino_lr, lr_image, task_id)

                pred = generate_and_decode(
                    model, latents_noise, cond_input,
                    latents_scale, latents_bias, args,
                    lr_image=lr_image, decoder=decoder
                )

                pred_gray = pred[:, 0:1]
                B, D, C, H, W = hr_image.shape
                gt_gray = hr_image.reshape(B * D, C, H, W)

                if not torch.isnan(pred_gray).any():
                    psnr_metric.update(pred_gray, gt_gray)
                    ssim_metric.update(pred_gray, gt_gray)

                if args.save_results:
                    all_preds.append(pred_gray.cpu().numpy())
                    all_gts.append(gt_gray.cpu().numpy())

        final_psnr = psnr_metric.compute().item()
        final_ssim = ssim_metric.compute().item()

        logger.info(f"[{task_id}] PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}")

        if args.save_results and all_preds:
            default_key = 'microdiffuse3d' if args.decoder_type == 'adapted' else 'microdiffuse3d_VAEdecoder'
            results_key = args.results_key or default_key
            all_preds = np.concatenate(all_preds, axis=0)
            all_gts = np.concatenate(all_gts, axis=0)

            with h5py.File(test_path, 'a') as hf:
                if results_key in hf:
                    del hf[results_key]
                hf.create_dataset(results_key, data=all_preds, compression='lzf')
            logger.info(f"Saved predictions ({all_preds.shape}) to '{results_key}' in {test_path}")


def main():
    parser = argparse.ArgumentParser(description="MicroDiffuse3D Evaluation")

    # Config files
    parser.add_argument("--task_configs_json", type=str, required=True)
    parser.add_argument("--backbone_args_json", type=str, required=True)
    parser.add_argument("--conditioning_args_json", type=str, required=True)

    # Checkpoint
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to a model state_dict (.pt) to load into JointModel.")

    # Sampling
    parser.add_argument("--num-steps", type=int, default=50, help="Number of diffusion sampling steps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--path-type", type=str, default="linear")

    # Decoder
    parser.add_argument("--decoder-type", type=str, default=None,
                        choices=[None, 'finetuned_vae', 'adapted'])
    parser.add_argument("--decoder-path", type=str, default=None)

    # Output
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--project-name", type=str, default="MicroDiffuse3D")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--results-key", type=str, default=None)

    # Model
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument('--z-types', nargs='+', default=[])

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
