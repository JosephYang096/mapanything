#!/usr/bin/env python
"""
把 wai_window3 训练的实际参数（已解析）更新到 WandB run，并上传原始 Hydra 配置。

为什么需要它：
- 训练启动时上传到 WandB 的 config 用了 resolve=False，插值(如 ${...})未解析成具体数值。
- 全量 OmegaConf.resolve=True 又会因配置里的 -inf 特殊浮点失败。
- 本脚本从 .hydra/config.yaml 直接按属性取值（OmegaConf 属性访问会自动解析插值），
  得到用户关心的具体参数，更新到 run 的 config，并把原始配置文件一并上传。

用法：
  source /mnt/workspace/yangyulong/code/mapanything/.venv/bin/activate
  export WANDB_API_KEY="<你的key>"
  python joseph_script/train/wandb_update_config.py [run_id]
"""
import sys
from pathlib import Path

from omegaconf import OmegaConf

import wandb

ROOT = Path("/mnt/workspace/yangyulong/code/mapanything")
RUN_DIR = ROOT / "map-anything-main/experiments/wai_window3_finetune"
ENTITY = "josephyang096-jd-com"
PROJECT = "map-anything"
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "20z0iwxc"
NUM_GPUS = 1  # wai_window3 单卡训练


def main():
    cfg = OmegaConf.load(RUN_DIR / ".hydra/config.yaml")
    tp = cfg.train_params
    ds = cfg.dataset

    sel = {}
    # ---- 损失函数设置 ----
    sel["loss.train_criterion"] = str(cfg.loss.train_criterion)
    sel["loss.test_criterion"] = str(cfg.loss.test_criterion)
    # ---- 学习率 / 训练超参数 ----
    sel["train_params.lr"] = tp.lr
    sel["train_params.min_lr"] = tp.min_lr
    sel["train_params.epochs"] = tp.epochs
    sel["train_params.warmup_epochs"] = tp.warmup_epochs
    sel["train_params.weight_decay"] = tp.weight_decay
    sel["train_params.schedule_type"] = tp.schedule_type
    sel["train_params.accum_iter"] = tp.accum_iter
    sel["train_params.amp"] = bool(tp.amp)
    sel["train_params.amp_dtype"] = tp.amp_dtype
    sel["train_params.print_freq"] = tp.print_freq
    sel["train_params.save_freq"] = tp.save_freq
    sel["train_params.keep_freq"] = tp.keep_freq
    sel["train_params.eval_freq"] = tp.eval_freq
    sel["train_params.resume"] = tp.resume
    # ---- 子模块（encoder）独立学习率 ----
    if tp.submodule_configs and "encoder" in tp.submodule_configs:
        enc = tp.submodule_configs.encoder
        sel["submodule.encoder.lr"] = enc.lr
        sel["submodule.encoder.min_lr"] = enc.min_lr
        sel["submodule.encoder.warmup_epochs"] = enc.warmup_epochs
        sel["submodule.encoder.weight_decay"] = enc.weight_decay
        sel["submodule.encoder.schedule_type"] = enc.schedule_type
    # ---- batch size ----
    sel["batch.max_num_of_imgs_per_gpu"] = int(tp.max_num_of_imgs_per_gpu)
    sel["batch.effective_batch_size"] = (
        NUM_GPUS * int(tp.max_num_of_imgs_per_gpu) / int(ds.num_views)
    )
    # ---- 模型 / 优化器 ----
    sel["model.model_str"] = str(cfg.model.model_str)
    if hasattr(cfg.model, "encoder"):
        sel["model.encoder.name"] = str(cfg.model.encoder.name)
    sel["model.pretrained"] = str(cfg.model.pretrained)
    sel["optimizer"] = "AdamW (betas=(0.9, 0.95))"

    # 更新到 WandB run
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
    run.config.update(sel)
    run.update()
    print(f">>> 已更新 run config ({len(sel)} 个字段) -> {ENTITY}/{PROJECT}/{RUN_ID}")
    for k, v in sel.items():
        print(f"    {k} = {v}")

    # 上传原始 Hydra 配置文件（完整备份）
    for f in ("config.yaml", "overrides.yaml", "hydra.yaml"):
        p = RUN_DIR / ".hydra" / f
        if p.exists():
            run.upload_file(str(p))
            print(">>> 已上传原始配置:", f)

    print(">>> 查看: https://wandb.ai/%s/%s/runs/%s" % (ENTITY, PROJECT, RUN_ID))


if __name__ == "__main__":
    main()
