"""Generate MapAnything dataset metadata (scene-list npy) for wai_window3.

Creates:
  <meta_root>/train/blendedmvs_scene_list_train.npy
  <meta_root>/val/blendedmvs_scene_list_val.npy
"""
import os
from pathlib import Path
import numpy as np

WAI_ROOT = Path("/mnt/workspace/yangyulong/code/mapanything/dataset/wai_window3")
META_ROOT = Path("/mnt/workspace/yangyulong/code/mapanything/dataset/wai_metadata")

scenes = sorted(p.name for p in WAI_ROOT.iterdir()
                if p.is_dir() and (p / "scene_meta.json").exists())
print(f"Found {len(scenes)} scenes")

# train = all but a small val holdout
rng = np.random.default_rng(0)
val = sorted(rng.choice(scenes, size=max(4, len(scenes) // 20), replace=False))
train = sorted(s for s in scenes if s not in val)

for split, lst in (("train", train), ("val", val)):
    out_dir = META_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.array(lst, dtype=object)
    np.save(out_dir / f"blendedmvs_scene_list_{split}.npy", arr)
    print(f"{split}: {len(lst)} scenes -> {out_dir / ('blendedmvs_scene_list_' + split + '.npy')}")

print("Done.")
