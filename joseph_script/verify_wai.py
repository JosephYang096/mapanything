"""Verify a converted WAI scene loads correctly via the WAI core API."""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
from mapanything.utils.wai.core import load_data, load_frame, get_intrinsics, get_extrinsics

import sys
scene_root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/workspace/yangyulong/code/mapanything/dataset/wai_window3/vid_20260603_143931"

meta = load_data(f"{scene_root}/scene_meta.json", "scene_meta")
print("scene_name       :", meta["scene_name"])
print("camera_model     :", meta["camera_model"])
print("camera_convention:", meta["camera_convention"])
print("shared_intrinsics:", meta["shared_intrinsics"])
print("n_frames         :", len(meta["frames"]))
print("frame_names[:3]  :", list(meta["frame_names"].items())[:3])
print("frame_modalities :", meta["frame_modalities"])

# Load a few frames and check geometry sanity.
for fi in [0, 37, 74]:
    fname = f"{fi:06d}"
    fr = load_frame(
        scene_root, fname,
        modalities=["image", "depth"],
        scene_meta=meta,
    )
    img = fr["image"]
    depth = fr["depth"]
    K = get_intrinsics(meta, fname)
    c2w = get_extrinsics(meta, fname)
    img_np = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    depth_np = depth.numpy() if hasattr(depth, "numpy") else np.asarray(depth)
    valid = depth_np[depth_np > 0]
    print(f"\n--- frame {fname} ---")
    print("  image shape :", tuple(img_np.shape), "dtype", img_np.dtype,
          "range", (float(img_np.min()), float(img_np.max())))
    print("  depth shape :", tuple(depth_np.shape), "dtype", depth_np.dtype)
    print("  depth valid : n=%d min=%.3f max=%.3f mean=%.3f" % (
        valid.size, float(valid.min()), float(valid.max()), float(valid.mean())))
    K_np = K.numpy() if hasattr(K, "numpy") else np.asarray(K)
    c2w_np = c2w.numpy() if hasattr(c2w, "numpy") else np.asarray(c2w)
    print("  K fx,fy,cx,cy: %.2f %.2f %.2f %.2f" % (
        K_np[0, 0], K_np[1, 1], K_np[0, 2], K_np[1, 2]))
    print("  c2w t (xyz) :", np.round(c2w_np[:3, 3], 4).tolist())
    R = c2w_np[:3, :3]
    print("  R det       : %.5f (orthonormal check |RtR-I|=%.2e)" % (
        np.linalg.det(R), np.abs(R.T @ R - np.eye(3)).max()))

print("\nOK: WAI scene verified.")