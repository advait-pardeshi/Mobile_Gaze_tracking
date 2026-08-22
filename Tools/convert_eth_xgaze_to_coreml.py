#!/usr/bin/env python3
"""
Convert hysts' pl_gaze_estimation ETH-XGaze ResNet18 checkpoint into a
CoreML model (`GazeNet.mlpackage`) suitable for iOS Stage 4 inference.

Assumes:
- timm + torch + coremltools are installed in the active Python.
- The .pth checkpoint lives at the path below (or pass --ckpt).

The wrapper module bakes ImageNet normalization
(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) into the network so
the iOS side can feed RGB floats in [0, 1] directly.

CoreML signature produced:
    input  : MLMultiArray  shape=[1, 3, 224, 224]  float32  (RGB, [0,1])
    output : MLMultiArray  shape=[1, 2]            float32  ((pitch, yaw) rad)

Usage:
    cd GazeTrackerPhase1/Tools
    python3 convert_eth_xgaze_to_coreml.py
"""

from __future__ import annotations
import argparse
import pathlib
import sys

import torch
import torch.nn as nn
import timm
import coremltools as ct

DEFAULT_CKPT = pathlib.Path(
    "/Users/advait/Desktop/GazeTracking/New Folder With Items/"
    "WebCamGazeEstimation-main/src/plgaze/models/eth-xgaze/"
    "eth-xgaze_resnet18.pth"
)
DEFAULT_OUTPUT = pathlib.Path(__file__).resolve().parent.parent / "Models" / "GazeNet.mlpackage"


class GazeNetWrapper(nn.Module):
    """ImageNet-normalize, then run the ETH-XGaze ResNet18."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224], RGB, in [0, 1]
        return self.model((x - self.mean) / self.std)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=pathlib.Path, default=DEFAULT_CKPT)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    if not args.ckpt.exists():
        print(f"[convert] ✗ checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[convert] loading checkpoint: {args.ckpt}")
    model = timm.create_model("resnet18", num_classes=2)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    wrapped = GazeNetWrapper(model).eval()

    print("[convert] tracing")
    example = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example)
        # Sanity check: trace should reproduce the eager output exactly.
        eager = wrapped(example)
        traced_out = traced(example)
        assert torch.allclose(eager, traced_out, atol=1e-5), \
            "traced output diverges from eager"
    print(f"[convert] sample output (pitch, yaw) for zero input: {eager[0].tolist()}")

    print("[convert] converting to CoreML")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, 3, 224, 224))],
        outputs=[ct.TensorType(name="gaze")],
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    mlmodel.short_description = (
        "ETH-XGaze ResNet18 (hysts/pl_gaze_estimation) — input: 1x3x224x224 "
        "RGB float [0,1]; output: (pitch, yaw) radians."
    )
    mlmodel.save(str(args.out))
    print(f"[convert] ✓ saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
