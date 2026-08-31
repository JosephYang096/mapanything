"""Scan all vid_* scenes for data completeness before conversion."""
import os
from pathlib import Path

root = Path("/mnt/workspace/yangyulong/code/mapanything/dataset/20260603_window_3")

scenes = sorted(d for d in os.listdir(root)
                if d.startswith("vid_") and (root / d).is_dir())

print(f"Total vid_* scenes: {len(scenes)}\n")

ok, bad = [], []
for s in scenes:
    sr = root / s
    # timestamp folders = those containing img.png
    ts = [d for d in os.listdir(sr)
          if(sr / d).is_dir() and (sr / d / "img.png").exists()]
    n_ts = len(ts)
    intr = sr / "intrinsics"
    extr = sr / "extrinsics"
    n_intr = len(list(intr.glob("*_meta.json"))) if intr.is_dir() else -1
    n_extr = len(list(extr.glob("*_extrinsic.json"))) if extr.is_dir() else -1

    # check per-index extrinsic files exist for 0..n_ts-1
    missing_extr = []
    if extr.is_dir():
        for i in range(n_ts):
            if not (extr / f"{i}_extrinsic.json").exists():
                missing_extr.append(i)
    else:
        missing_extr = list(range(n_ts))

    status = "OK"
    if n_ts != n_intr or n_ts != n_extr or missing_extr:
        status = "BAD"
        bad.append((s, n_ts, n_intr, n_extr, len(missing_extr)))
    else:
        ok.append(s)
    if status == "BAD":
        print(f"[BAD] {s}: ts={n_ts} intr={n_intr} extr={n_extr} missing_extr={len(missing_extr)}")

print(f"\nSummary: OK={len(ok)}  BAD={len(bad)}")
if bad:
    print("BAD scenes:", [b[0] for b in bad])