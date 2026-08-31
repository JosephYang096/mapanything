#!/bin/bash
# =============================================================
# wai_window3 训练启动 + 自动同步 WandB（纯 joseph_script，不改框架代码）
# 训练指标通过 wandb sync 持续上传到 WandB，无需看本地 TensorBoard。
#
# 用法：
#   export WANDB_API_KEY="<你的key>"
#   bash joseph_script/train/run_train_wandb.sh [--no-sync]
# =============================================================
set -e
ROOT=/mnt/workspace/yangyulong/code/mapanything
MAIN=$ROOT/map-anything-main
RUN_DIR=$MAIN/experiments/wai_window3_finetune
LOG=$ROOT/train_wai.log
CLEAN_CFG=$ROOT/joseph_script/train/wandb_config_clean.yaml
PROJECT=${WANDB_PROJECT:-map-anything}
SYNC_INTERVAL=300
DO_SYNC=1
[ "$1" = "--no-sync" ] && DO_SYNC=0

if [ -z "$WANDB_API_KEY" ]; then
  echo "警告：未设置 WANDB_API_KEY，将使用已保存的 wandb 登录凭据"
fi

source "$ROOT/.venv/bin/activate"
export HYDRA_FULL_ERROR=1
cd "$MAIN"

# 1) 启动训练（后台，脱离会话防 SIGHUP）
setsid nohup torchrun --nproc_per_node 1 \
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
    model.pretrained="$MAIN/checkpoints/map-anything.pth" \
    train_params=lower_encoder_lr \
    train_params.epochs=50 \
    train_params.warmup_epochs=2 \
    train_params.keep_freq=5 \
    train_params.max_num_of_imgs_per_gpu=8 \
    hydra.run.dir="$RUN_DIR" \
    > "$LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
echo "TRAIN PID=$TRAIN_PID  (日志: $LOG)"

# 2) 后台定期同步 TensorBoard -> WandB（用户只看 WandB）
if [ "$DO_SYNC" = "1" ]; then
  (
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
      wandb sync --sync-tensorboard --project "$PROJECT" \
        --config "$CLEAN_CFG" "$RUN_DIR" >/dev/null 2>&1 || true
      sleep "$SYNC_INTERVAL"
    done
    # 训练结束后再同步一次完整数据
    wandb sync --sync-tensorboard --project "$PROJECT" \
      --config "$CLEAN_CFG" "$RUN_DIR" >/dev/null 2>&1 || true
  ) &
  echo "WANDB-SYNC PID=$!  (每 ${SYNC_INTERVAL}s 同步一次, project=$PROJECT)"
else
  echo "WANDB-SYNC 已禁用 (--no-sync)"
fi
