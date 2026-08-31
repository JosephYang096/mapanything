"""
Compute pairwise covisibility maps (WAI `covisibility/v0`) for every scene under a WAI root,
using the ground-truth depth that already exists in the converted dataset.

This replicates the "full" denominator mode of MapAnything's official covisibility stage:
  covisibility[i, j] = #(pixels of view i that reproject consistently into view j) / (H*W)

Output per scene:  <root>/<scene>/covisibility/v0/pairwise_covisibility--<N>x<N>.npy
"""
import os
import sys
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import torch
import torch.nn.functional as F

from mapanything.utils.wai.core import load_data, load_frame, get_intrinsics, get_extrinsics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = 224          # long side of depth maps used for covis
DEPTH_ASSOC_ERR = 0.1      # absolute depth error threshold (m)
DEPTH_ASSOC_REL = 0.005    # relative depth error threshold
DEPTH_ASSOC_TEMP = 0.1
FRUSTUM_CHECK = True
CHUNK = 4000


def _resize_depth(depth, long_side):
    h, w = depth.shape
    scale = long_side / max(h, w)
    if abs(scale - 1.0) < 1e-4:
        return depth, h, w
    new_h = int(round(h * scale / 4) * 4)
    new_w = int(round(w * scale / 4) * 4)
    d = torch.from_numpy(depth).float().to(DEVICE)[None, None]
    d = F.interpolate(d, size=(new_h, new_w), mode="bilinear", align_corners=False)[0, 0]
    return d, new_h, new_w


def load_scene(scene_root: Path):
    meta = load_data(scene_root / "scene_meta.json", "scene_meta")
    names = sorted(meta["frames"], key=lambda f: f["frame_name"])
    depths, valid, Ks, c2ws = [], [], [], []
    for frame in names:
        fr = load_frame(scene_root, frame["frame_name"], modalities=["depth"], scene_meta=meta)
        d = fr["depth"].numpy().astype(np.float32)
        K = get_intrinsics(meta, frame["frame_name"]).numpy().astype(np.float32)
        c2w = get_extrinsics(meta, frame["frame_name"]).numpy().astype(np.float32)
        depths.append(d)
        Ks.append(K)
        c2ws.append(c2w)
    return meta, depths, Ks, c2ws


def frustum_intersection(depths, valid, Ks, c2ws):
    """Conservative AABB overlap of view frustum 8-corner hulls (pre-filter only)."""
    N = len(depths)
    corners = torch.tensor(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
        dtype=torch.float32, device=DEVICE,
    )  # normalized device coords, z=0 near / z=1 far
    near = torch.stack([d[v].min() if v.any() else torch.tensor(0.0) for d, v in zip(depths, valid)]).to(DEVICE)
    far = torch.stack([d[v].max() if v.any() else torch.tensor(0.0) for d, v in zip(depths, valid)]).to(DEVICE)

    frusts = []
    for K, c2w, n, f in zip(Ks, c2ws, near, far):
        Kt = torch.tensor(K, device=DEVICE)
        R = torch.tensor(c2w[:3, :3], device=DEVICE)
        t = torch.tensor(c2w[:3, 3], device=DEVICE)
        pts_cam = []
        for z in (n, f):
            # camera coords: x = u*z, y = v*z (opencv: y down), z forward
            uc = corners[:, 0] * z
            vc = corners[:, 1] * z
            pts_cam.append(torch.stack([uc, vc, torch.full_like(uc, z), torch.ones_like(uc)], dim=-1))
        pts_cam = torch.cat(pts_cam, dim=0)  # (16, 4)
        pts_world = pts_cam[:, :3] @ R.T + t
        frusts.append(pts_world)
    frusts = torch.stack(frusts)  # (N, 16, 3)

    mins = frusts.min(dim=1).values
    maxs = frusts.max(dim=1).values
    inter = ((mins[:, None] <= maxs[None, :]).all(dim=-1) &
             (maxs[:, None] >= mins[None, :]).all(dim=-1))
    return inter


def compute_scene_covis(scene_root: Path, overwrite: bool = False):
    meta, depths, Ks, c2ws = load_scene(scene_root)
    N = len(depths)
    if N == 0:
        return 0

    out_dir = scene_root / "covisibility" / "v0"
    out_dir.mkdir(parents=True, exist_ok=True)

    # resize depth maps to TARGET_SIZE (need all same resolution)
    resized = [_resize_depth(d, TARGET_SIZE) for d in depths]
    depths_r = torch.stack([r[0] for r in resized]).to(DEVICE)
    dh, dw = resized[0][1], resized[0][2]
    scale_h = dh / 480.0
    scale_w = dw / 640.0
    Ks_r = torch.stack([torch.tensor(K, device=DEVICE) for K in Ks])
    Ks_r = Ks_r.clone()
    Ks_r[:, 0] *= scale_w
    Ks_r[:, 1] *= scale_h
    valid_r = depths_r > 0
    c2ws_t = torch.stack([torch.tensor(c, device=DEVICE) for c in c2ws])

    # world points for each view (N, H, W, 3)
    ys, xs = torch.meshgrid(torch.arange(dh, device=DEVICE), torch.arange(dw, device=DEVICE), indexing="ij")
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=-1).float()  # (H, W, 3)
    invK = torch.inverse(Ks_r)
    world_pts = []
    for i in range(N):
        Kinv = invK[i]
        cam_pts = (pix @ Kinv.T) * depths_r[i, :, :, None]
        world_pts.append((cam_pts @ c2ws_t[i, :3, :3].T) + c2ws_t[i, :3, 3])
    world_pts = torch.stack(world_pts)  # (N, H, W, 3)

    # optional frustum pre-filter
    frust = None
    if FRUSTUM_CHECK:
        try:
            frust = frustum_intersection(depths_r, valid_r, Ks_r, c2ws_t)
        except Exception:
            frust = None

    cov = torch.zeros((N, N), device="cpu")
    for i in range(N):
        if frust is not None:
            ov = frust[i].nonzero().squeeze(1)
        else:
            ov = torch.arange(N, device=DEVICE)
        if len(ov) == 0:
            continue
        score = torch.zeros(N, device="cpu")
        # points of view i in world
        pts_i = world_pts[i].reshape(-1, 3)  # (H*W, 3)
        valid_i = valid_r[i].reshape(-1)
        for s in range(0, len(ov), CHUNK):
            chunk = ov[s:s + CHUNK]
            # camera coords of pts_i in each target view
            R = c2ws_t[chunk, :3, :3]  # (V,3,3)
            t = c2ws_t[chunk, :3, 3]   # (V,3)
            cam = (pts_i[None] - t[:, None]) @ R  # (V, HW, 3), camera coords of target view
            z = cam[..., 2]
            # project to target-view pixel coords (fx*zx+cx etc.)
            fx = Ks_r[chunk, 0, 0].unsqueeze(1)
            fy = Ks_r[chunk, 1, 1].unsqueeze(1)
            cx = Ks_r[chunk, 0, 2].unsqueeze(1)
            cy = Ks_r[chunk, 1, 2].unsqueeze(1)
            u = fx * cam[..., 0] / z.clamp(min=1e-6) + cx
            v = fy * cam[..., 1] / z.clamp(min=1e-6) + cy
            in_img = (u >= 0) & (u < dw) & (v >= 0) & (v < dh) & (z > 0.04) & valid_i[None]
            # sample expected depth in target views (gather to avoid broadcast)
            uu = u.clamp(0, dw - 1).long()
            vv = v.clamp(0, dh - 1).long()
            lin = (vv * dw + uu).reshape(len(chunk), -1)
            depth_t = depths_r[chunk].flatten(1).gather(1, lin)
            err = torch.abs(depth_t - z)
            thres = DEPTH_ASSOC_ERR + DEPTH_ASSOC_REL * z - np.log(0.5) * DEPTH_ASSOC_TEMP
            match = (err < thres) & in_img
            score[chunk.cpu()] = (match.sum(dim=1) / (dh * dw)).clamp(0, 1).cpu()
        cov[i] = score
        torch.cuda.empty_cache()

    # store as float32 memmap .npy with WAI naming convention
    cov = cov.numpy().astype(np.float32)
    shape_str = "x".join(str(dim) for dim in cov.shape)
    mmap_name = f"pairwise_covisibility--{shape_str}.npy"
    with open(out_dir / mmap_name, "wb") as fid:
        np.save(fid, cov)
    return N


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/workspace/yangyulong/code/mapanything/dataset/wai_window3")
    only = sys.argv[2] if len(sys.argv) > 2 else None
    overwrite = "--overwrite" in sys.argv
    scenes = sorted(p for p in root.iterdir() if p.is_dir() and (p / "scene_meta.json").exists())
    if only:
        scenes = [p for p in scenes if p.name == only]
    print(f"Computing covisibility for {len(scenes)} scenes under {root} ...")
    for i, sc in enumerate(scenes):
        out = sc / "covisibility" / "v0"
        if out.exists() and not overwrite:
            print(f"[skip] {sc.name}: covisibility exists (use --overwrite)")
            continue
        try:
            n = compute_scene_covis(sc, overwrite=overwrite)
            print(f"[done {i+1}/{len(scenes)}] {sc.name}: {n} frames -> {out}")
        except Exception as e:
            print(f"[FAIL] {sc.name}: {type(e).__name__}: {e}")
    print("Finished.")


if __name__ == "__main__":
    main()
