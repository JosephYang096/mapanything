"""Probe: build MapAnything model and check pretrained checkpoint alignment."""
import os
import sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import torch
from mapanything.models import init_model

# Build configs manually to mirror hydra resolution
import hydra
from omegaconf import OmegaConf

CKPT = "/mnt/workspace/yangyulong/code/mapanything/map-anything-main/checkpoints/map-anything.pth"


def main():
    # load via hydra compose so defaults resolve
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir="/mnt/workspace/yangyulong/code/mapanything/map-anything-main/configs",
    ):
        cfg = hydra.compose(
            config_name="train",
            overrides=[
                "machine=default",
                "model=mapanything",
                "model.encoder.uses_torch_hub=false",
                "model.task=images_only",
            ],
        )

    model_str = cfg.model.model_str
    model_config = cfg.model.model_config
    model = init_model(model_str, model_config, torch_hub_force_reload=False)
    print("Model built:", type(model).__name__)
    print("scene_rep_type:", getattr(model, "scene_rep_type", None))
    print("info_sharing module args name:", model.info_sharing.name)
    print("n params (M):", sum(p.numel() for p in model.parameters()) / 1e6)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing keys:", len(missing))
    for k in list(missing)[:10]:
        print("   MISSING:", k)
    print("unexpected keys:", len(unexpected))
    for k in list(unexpected)[:10]:
        print("   UNEXPECTED:", k)

    # Count actual loaded params
    own = set(model.state_dict().keys())
    total_missing = len(missing)
    print("=> total missing:", total_missing)


if __name__ == "__main__":
    main()
