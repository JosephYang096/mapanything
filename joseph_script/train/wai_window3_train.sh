#!/bin/bash
# Fine-tune MapAnything on the custom wai_window3 dataset (single GPU L20).
# Based on the official example: bash_scripts/train/examples/mapa_img_only_4v_bmvs_48ipg_8g.sh
# Changes vs official: machine=custom (paths), dataset=custom_window3 (our data),
#   model.pretrained=<our map-anything.pth> (load pretrained weights), single GPU.

NUM_GPUS=${1:-1}
export HYDRA_FULL_ERROR=1

source /mnt/workspace/yangyulong/code/mapanything/.venv/bin/activate
cd /mnt/workspace/yangyulong/code/mapanything/map-anything-main

torchrun --nproc_per_node ${NUM_GPUS} \
    scripts/train.py \
    machine=custom \
    dataset=custom_window3 dataset.num_workers=8 \
    dataset.num_views=2 \
    loss=overall_loss_highpm_plus_rel_pose \
    model=mapanything \
    model/task=images_only \
    model.encoder.uses_torch_hub=false \
    model.encoder.gradient_checkpointing=true \
    model.info_sharing.module_args.gradient_checkpointing=true \
    model.pretrained=/mnt/workspace/yangyulong/code/mapanything/map-anything-main/checkpoints/map-anything.pth \
    train_params=lower_encoder_lr \
    train_params.epochs=50 \
    train_params.warmup_epochs=2 \
    train_params.keep_freq=5 \
    train_params.max_num_of_imgs_per_gpu=8 \
    hydra.run.dir='${root_experiments_dir}/wai_window3_finetune'
