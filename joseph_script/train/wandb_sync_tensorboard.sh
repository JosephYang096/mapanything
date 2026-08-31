#!/bin/bash
# =============================================================
# wai_window3 训练 -> WandB 上传工具（纯 joseph_script，不改框架代码）
#
# 原理：训练仍用 map-anything-main 框架自带的 TensorBoard 记录指标，
#       本脚本在训练结束后，把 TensorBoard 日志目录（tfevents + .hydra 配置）
#       同步上传到 WandB，实现"训练过程 + 配置"的记录。
#
# 用法：
#   export WANDB_API_KEY="<你的key>"     # 未登录过则必须设置
#   bash /mnt/workspace/yangyulong/code/mapanything/joseph_script/train/wandb_sync_tensorboard.sh
#   # 可选覆盖：WANDB_PROJECT=xxx bash wandb_sync_tensorboard.sh
# =============================================================
set -e

RUN_DIR="/mnt/workspace/yangyulong/code/mapanything/map-anything-main/experiments/wai_window3_finetune"
PROJECT="${WANDB_PROJECT:-map-anything}"

if [ ! -d "$RUN_DIR" ] || [ ! -f "$RUN_DIR/.hydra/config.yaml" ]; then
  echo "错误：找不到训练输出目录 $RUN_DIR（或缺少 .hydra/config.yaml）" >&2
  exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
  echo "提示：未检测到 WANDB_API_KEY 环境变量，将尝试使用已保存的 wandb 登录凭据..."
fi

cd /mnt/workspace/yangyulong/code/mapanything/map-anything-main || exit 1

echo ">>> 同步 TensorBoard 日志到 WandB (project=$PROJECT)"
echo ">>> 来源目录: $RUN_DIR"
wandb sync --sync-tensorboard --project "$PROJECT" \
  --config "$RUN_DIR/.hydra/config.yaml" \
  "$RUN_DIR"

echo ">>> 完成。查看: https://wandb.ai/$PROJECT"
