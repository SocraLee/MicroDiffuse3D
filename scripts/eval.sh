#!/bin/bash
# MicroDiffuse3D Evaluation Script

# ==========================================
# Generation with Adapted post-diffusion decoder (Recommended)
# ==========================================
CUDA_VISIBLE_DEVICES=0 python microdiffuse3d/generate.py \
    --task_configs_json "./configs/3dsr4z_config.json" \
    --backbone_args_json "./configs/backbone_config.json" \
    --conditioning_args_json "./configs/encoder_config.json" \
    --ckpt_path "./outputs/3dsr4z_experiment/checkpoints/acc_step_0040000/model.pt" \
    --exp-name "3dsr4z_eval_adapted" \
    --save-results \
    --batch-size 4 \
    --num-steps 50 \
    --decoder-type adapted \
    --decoder-path "./outputs/decoder/adapted/3d_sr/best_decoder.pth" \
    --results-key microdiffuse3d


# ==========================================
# Generation with VAE decoder
# ==========================================
CUDA_VISIBLE_DEVICES=0 python microdiffuse3d/generate.py \
    --task_configs_json "./configs/3dsr4z_config.json" \
    --backbone_args_json "./configs/backbone_config.json" \
    --conditioning_args_json "./configs/encoder_config.json" \
    --ckpt_path "./outputs/3dsr4z_experiment/checkpoints/acc_step_0040000/model.pt" \
    --exp-name "3dsr4z_eval_40k" \
    --save-results \
    --batch-size 4 \
    --num-steps 50 \
    --project-name "MicroDiffuse3D" \
    --results-key microdiffuse3d_VAEdecoder