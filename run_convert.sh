#!/bin/bash
# Robust background launcher for the WAI conversion (survives SSH drops).
set -e
cd /mnt/workspace/yangyulong/code/mapanything
source .venv/bin/activate
cd map-anything-main/data_processing

LOG=/mnt/workspace/yangyulong/code/mapanything/convert_all.log
: > "$LOG"

# -u : unbuffered stdout so progress is visible in the log immediately.
setsid python -u convert_custom_to_wai.py \
  --original_root \
    /mnt/workspace/yangyulong/code/mapanything/dataset/20260603_window_3 \
    /mnt/workspace/yangyulong/code/mapanything/dataset/20260626_window_3 \
    /mnt/workspace/yangyulong/code/mapanything/dataset/20260727_window_3 \
  --root /mnt/workspace/yangyulong/code/mapanything/dataset/wai_window3 \
  --overwrite \
  >> "$LOG" 2>&1 < /dev/null &

echo "PID=$!"