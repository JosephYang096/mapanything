"""Patch each wai_window3 scene_meta.json so the standard BlendedMVSWAI loader can
consume our GT `mask` as the non-ambiguous mask modality (`pred_mask/moge2`)."""
import json
from pathlib import Path

WAI_ROOT = Path("/mnt/workspace/yangyulong/code/mapanything/dataset/wai_window3")

patched = 0
for p in sorted(WAI_ROOT.iterdir()):
    if not p.is_dir():
        continue
    meta_path = p / "scene_meta.json"
    if not meta_path.exists():
        continue
    meta = json.loads(meta_path.read_text())
    fmods = meta.get("frame_modalities", {})
    if "mask" not in fmods:
        continue
    if "pred_mask" in fmods:
        continue
    fmods["pred_mask"] = {
        "moge2": {"frame_key": "mask", "format": "binary"}
    }
    meta["frame_modalities"] = fmods
    meta_path.write_text(json.dumps(meta, indent=2))
    patched += 1

print(f"Patched {patched} scene_meta.json files")
