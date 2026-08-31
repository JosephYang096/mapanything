#!/usr/bin/env python
"""
把 WandB run 的 config【完全清空并重建】为干净、已解析的关键训练参数。

解决：训练启动时上传的 config 是 resolve=False 生成的，
残留了大量深层嵌套(model/dataset/machine/pred_head 内部参数)与未解析插值，
在 WandB UI 上既难读又没价值。

本脚本只保留清晰有用的关键参数（损失/学习率/batchsize/调度/数据集/模型等），
并删除其余所有杂项，最终 config 干净可读。

用法：
  source /mnt/workspace/yangyulong/code/mapanything/.venv/bin/activate
  export WANDB_API_KEY="<你的key>"
  python joseph_script/train/wandb_replace_config.py [run_id]
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
NUM_GPUS = 1


def build_clean_config():
    cfg = OmegaConf.load(RUN_DIR / ".hydra/config.yaml")
    tp = cfg.train_params
    ds = cfg.dataset
    sel = {}
    sel["loss.train_criterion"] = str(cfg.loss.train_criterion)
    sel["loss.test_criterion"] = str(cfg.loss.test_criterion)
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
    if tp.submodule_configs and "encoder" in tp.submodule_configs:
        enc = tp.submodule_configs.encoder
        sel["submodule.encoder.lr"] = enc.lr
        sel["submodule.encoder.min_lr"] = enc.min_lr
        sel["submodule.encoder.warmup_epochs"] = enc.warmup_epochs
        sel["submodule.encoder.weight_decay"] = enc.weight_decay
        sel["submodule.encoder.schedule_type"] = enc.schedule_type
    sel["batch.max_num_of_imgs_per_gpu"] = int(tp.max_num_of_imgs_per_gpu)
    sel["batch.effective_batch_size"] = (
        NUM_GPUS * int(tp.max_num_of_imgs_per_gpu) / int(ds.num_views)
    )
    sel["model.model_str"] = str(cfg.model.model_str)
    if hasattr(cfg.model, "encoder"):
        sel["model.encoder.name"] = str(cfg.model.encoder.name)
    sel["model.pretrained"] = str(cfg.model.pretrained)
    sel["optimizer"] = "AdamW (betas=(0.9, 0.95))"
    return sel


def main():
    sel = build_clean_config()
    keep = set(sel.keys())

    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

    # 1) 删除所有不在精选列表中的旧字段（巨型嵌套 + 未解析项）
    removed = []
    for k in list(run.config.keys()):
        if k not in keep:
            try:
                del run.config[k]
                removed.append(k)
            except Exception as e:  # 某些版本只读时降级处理
                run.config[k] = None
                removed.append(f"{k}(->None)")
    # 2) 写入干净的精选字段
    run.config.update(sel)
    # 3) 推送
    run.update()

    # 4) 重新拉取验证
    run2 = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
    final = dict(run2.config)
    print(">>> 已删除旧字段数:", len(removed))
    print(">>> 最终 config 字段数:", len(final))
    unresolved = [k for k, v in final.items() if isinstance(v, str) and "${" in v]
    print(">>> 未解析残留:", unresolved if unresolved else "无")
    for k in sorted(final.keys()):
        print(f"    {k} = {final[k]}")
    # 5) 上传原始 Hydra 配置到 Files 供完整参考
    for f in ("config.yaml", "overrides.yaml", "hydra.yaml"):
        p = RUN_DIR / ".hydra" / f
        if p.exists():
            run.upload_file(str(p))
    print(">>> 查看: https://wandb.ai/%s/%s/runs/%s" % (ENTITY, PROJECT, RUN_ID))


if __name__ == "__main__":
    main()
