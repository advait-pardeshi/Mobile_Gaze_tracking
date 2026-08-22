#!/usr/bin/env python3
"""
Fine-tune the ETH-XGaze ResNet18 on per-user data collected by the iOS
app's "Collect Fine-tune Data" flow.

Input: a `.zip` bundle produced by `FineTuneDataCollector.finalize()`,
       containing:
         - frames/<trial>_<frame>.png   (224×224 RGB face crops)
         - labels.csv                   (one row per frame: see schema in
                                         FineTuneDataCollector.swift)
         - meta.json                    (intrinsics, screen size, focal_norm)
         - calibration.json             (CalibrationModel: translation,
                                         targets, offsets — all in screen
                                         points, center-relative)

Pseudo-ground-truth label derivation (per frame):
  1. Cell center in screen-center-relative points:
         Sg = (cell_cx - screen_w/2, cell_cy - screen_h/2)
  2. CalibrationModel.predict() in iOS is a 50/50 mix of
         pGlobal       = (tx + tz·gx/gz, ty + tz·gy/gz)
         pLocal[i]     = pGlobal + offset[nearest target i to pGlobal]
     i.e. result = pGlobal + 0.5·offset[i*]. Invert by:
       a) ignore local correction in the first pass — solve
              tx + tz·gx/gz = Sg.x
              ty + tz·gy/gz = Sg.y
          for (gx/gz, gy/gz) given the fitted t = (tx, ty, tz). This yields
          the gaze direction (gx, gy, gz) ∝ (gx/gz, gy/gz, 1), normalised.
       b) refine: pick nearest target i to that pGlobal, then re-solve with
              Sg' = Sg - 0.5·offset[i*]  →  gives final (gx, gy, gz).
  3. Transform to normalized space:  gaze_norm = R_n · gaze_cam
  4. ETH-XGaze convention (see GazeEstimator.swift):
         g_eth = -gaze_norm
         pitch = asin(g_eth.y)
         yaw   = atan2(g_eth.x, g_eth.z)
  5. The FaceNormalizer H-flips the image before inference, and
     GazeEstimator negates the predicted yaw to undo that flip. So at
     training time, after flipping the loaded PNG horizontally (which we
     do via transforms.RandomHorizontalFlip(p=1.0)? — no, we don't flip
     again, the PNG is already the post-flip image the CNN consumed), the
     yaw label must match the CNN's frame, i.e. yaw is already correct.

Usage:
    python3 finetune_eth_xgaze.py \
        --bundle ~/Downloads/finetune_20260601_120000.zip \
        --base-checkpoint /path/to/eth-xgaze_resnet18.pth \
        --out-checkpoint  /path/to/eth-xgaze_resnet18_ft.pth \
        --epochs 5 --lr 1e-4 --batch 32

Then re-run `convert_eth_xgaze_to_coreml.py` with `--ckpt <out>` to
produce a fresh `Models/GazeNet.mlpackage`.
"""

from __future__ import annotations
import argparse
import csv
import io
import json
import math
import pathlib
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
import timm


# ----------------------------------------------------------------------
# Bundle loading
# ----------------------------------------------------------------------

@dataclass
class Bundle:
    root: pathlib.Path
    meta: dict
    calibration: dict
    labels: list[dict]   # one dict per labels.csv row (str values)

    @property
    def screen_w(self) -> float:
        return float(self.meta["screen_w_pt"])

    @property
    def screen_h(self) -> float:
        return float(self.meta["screen_h_pt"])

    @property
    def calibration_t(self) -> np.ndarray:
        return np.array(self.calibration["translation"], dtype=np.float64)

    @property
    def calibration_targets(self) -> np.ndarray:
        return np.array(self.calibration["targets"], dtype=np.float64)

    @property
    def calibration_offsets(self) -> np.ndarray:
        return np.array(self.calibration["offsets"], dtype=np.float64)


def load_bundle(zip_path: pathlib.Path) -> Bundle:
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="ftbundle_"))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmpdir)
    # The zip wraps a single run-named directory.
    children = [p for p in tmpdir.iterdir() if p.is_dir()]
    root = children[0] if len(children) == 1 else tmpdir
    with (root / "meta.json").open() as f:
        meta = json.load(f)
    with (root / "calibration.json").open() as f:
        calibration = json.load(f)
    rows: list[dict] = []
    with (root / "labels.csv").open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return Bundle(root=root, meta=meta, calibration=calibration, labels=rows)


# ----------------------------------------------------------------------
# Label derivation
# ----------------------------------------------------------------------

def invert_screen_mapper(target_screen_relative: np.ndarray,
                         t: np.ndarray) -> np.ndarray:
    """
    Given screen-center-relative target Sg = (sx, sy) and the fitted
    `t = (tx, ty, tz)`, solve for the gaze ratios (gx/gz, gy/gz) such that
        sx = tx + tz · gx/gz
        sy = ty + tz · gy/gz
    Returns the unit camera-frame gaze direction (gx, gy, gz).

    The forward projection is ratio-based, so g and -g both satisfy it.
    The physical solution has gz < 0: the user sits in front of the
    camera (+z in camera coords) and gazes back toward the screen plane
    (-z direction). We pick that branch explicitly — otherwise every
    label is 180° flipped and pre-tune angular error sits around 176°.
    """
    tx, ty, tz = t
    if abs(tz) < 1e-6:
        return np.array([0.0, 0.0, -1.0])
    rx = (target_screen_relative[0] - tx) / tz
    ry = (target_screen_relative[1] - ty) / tz
    g = np.array([rx, ry, 1.0])
    g = g / np.linalg.norm(g)
    if g[2] > 0:
        g = -g
    return g


def derive_pitch_yaw(row: dict, bundle: Bundle) -> Optional[tuple[float, float]]:
    """Return (pitch, yaw) in radians for one labels.csv row, or None."""
    try:
        cell_cx = float(row["cell_cx_pt"])
        cell_cy = float(row["cell_cy_pt"])
    except (KeyError, ValueError):
        return None
    Sg = np.array([cell_cx - bundle.screen_w / 2.0,
                   cell_cy - bundle.screen_h / 2.0])

    t = bundle.calibration_t
    targets = bundle.calibration_targets
    offsets = bundle.calibration_offsets

    # First pass: global-only inversion to figure out which target is
    # "nearest" under the same rule iOS uses.
    g_cam = invert_screen_mapper(Sg, t)
    # g_cam is unit-norm with non-zero z (invert_screen_mapper guarantees
    # gz ≈ -1 unless tz is degenerate), so direct division is safe.
    p_global = np.array([
        t[0] + t[2] * g_cam[0] / g_cam[2],
        t[1] + t[2] * g_cam[1] / g_cam[2],
    ])
    dists = np.linalg.norm(targets - p_global, axis=1)
    nearest = int(np.argmin(dists))
    # iOS averages pGlobal with pGlobal + offset[i*], so the realised
    # prediction is pGlobal + 0.5·offset[i*]. To make our derived gaze
    # produce the cell center, pre-subtract 0.5·offset from Sg and invert
    # again.
    Sg_corr = Sg - 0.5 * offsets[nearest]
    g_cam = invert_screen_mapper(Sg_corr, t)

    # R_n is row-major from the CSV (Rn_rc), encode back as a 3x3.
    try:
        Rn = np.array([
            [float(row["Rn_00"]), float(row["Rn_01"]), float(row["Rn_02"])],
            [float(row["Rn_10"]), float(row["Rn_11"]), float(row["Rn_12"])],
            [float(row["Rn_20"]), float(row["Rn_21"]), float(row["Rn_22"])],
        ])
    except (KeyError, ValueError):
        return None
    gaze_norm = Rn @ g_cam

    # ETH-XGaze: g_eth = -gaze_norm; pitch = asin(g_eth.y); yaw = atan2(g_eth.x, g_eth.z)
    g_eth = -gaze_norm
    g_eth /= max(np.linalg.norm(g_eth), 1e-9)
    pitch = math.asin(max(-1.0, min(1.0, float(g_eth[1]))))
    yaw = math.atan2(float(g_eth[0]), float(g_eth[2]))

    # FaceNormalizer flipped the image horizontally before the CNN saw
    # it. GazeEstimator undoes that on the *predicted* yaw by negating the
    # X-component before applying R_n^T. To stay consistent we negate the
    # yaw label here so the CNN learns to produce the same convention
    # that GazeEstimator post-processes.
    yaw = -yaw

    if not (math.isfinite(pitch) and math.isfinite(yaw)):
        return None
    return pitch, yaw


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FineTuneDataset(Dataset):
    def __init__(self, bundle: Bundle):
        self.bundle = bundle
        self.entries: list[tuple[str, float, float]] = []
        for row in bundle.labels:
            pj = derive_pitch_yaw(row, bundle)
            if pj is None:
                continue
            png_name = row.get("png_filename")
            if not png_name:
                continue
            png_path = bundle.root / "frames" / png_name
            if not png_path.exists():
                continue
            self.entries.append((str(png_path), pj[0], pj[1]))
        if not self.entries:
            raise RuntimeError("No usable labelled frames found in bundle.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, pitch, yaw = self.entries[idx]
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0          # H, W, 3
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = np.transpose(arr, (2, 0, 1))                       # 3, H, W
        return (
            torch.from_numpy(arr).float(),
            torch.tensor([pitch, yaw], dtype=torch.float32),
        )


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------

def angular_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean angular error in degrees between two batches of (pitch, yaw)."""
    def to_xyz(t: torch.Tensor) -> torch.Tensor:
        p = t[:, 0]
        y = t[:, 1]
        return torch.stack([torch.cos(p) * torch.sin(y),
                            torch.sin(p),
                            torch.cos(p) * torch.cos(y)], dim=1)
    a = to_xyz(pred)
    b = to_xyz(target)
    cos = (a * b).sum(dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos).mean() * (180.0 / math.pi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=pathlib.Path, required=True)
    ap.add_argument("--base-checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--out-checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--freeze-stem", action="store_true",
                    help="Freeze conv1 + bn1 to avoid catastrophic drift.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[finetune] loading bundle: {args.bundle}")
    bundle = load_bundle(args.bundle)
    print(f"[finetune] meta:        screen {bundle.screen_w:.0f} × {bundle.screen_h:.0f}")
    print(f"[finetune] calibration: t = {bundle.calibration_t.tolist()}")
    print(f"[finetune] {len(bundle.labels)} rows in labels.csv")

    ds = FineTuneDataset(bundle)
    n_val = max(1, int(round(len(ds) * args.val_split)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    print(f"[finetune] dataset: {len(ds)} usable frames "
          f"({n_train} train / {n_val} val)")

    model = timm.create_model("resnet18", num_classes=2)
    ckpt = torch.load(args.base_checkpoint, map_location="cpu",
                      weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)

    if args.freeze_stem:
        for p in model.conv1.parameters():
            p.requires_grad = False
        for p in model.bn1.parameters():
            p.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available()
                                else "cpu"))
    print(f"[finetune] device: {device}")
    model = model.to(device)

    @torch.no_grad()
    def eval_loop() -> tuple[float, float]:
        model.eval()
        losses, errs = [], []
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            p = model(x)
            losses.append(((p - y) ** 2).mean().item())
            errs.append(angular_error(p, y).item())
        return float(np.mean(losses)), float(np.mean(errs))

    pre_loss, pre_err = eval_loop()
    print(f"[finetune] pre-tune  val MSE={pre_loss:.5f}  ang_err={pre_err:.3f}°")

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        t_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            p = model(x)
            loss = ((p - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            t_losses.append(loss.item())
        v_loss, v_err = eval_loop()
        print(f"[finetune] epoch {epoch + 1}/{args.epochs}  "
              f"train MSE={np.mean(t_losses):.5f}  "
              f"val MSE={v_loss:.5f}  ang_err={v_err:.3f}°")

    post_loss, post_err = eval_loop()
    print(f"[finetune] post-tune val MSE={post_loss:.5f}  ang_err={post_err:.3f}°  "
          f"(Δ {post_err - pre_err:+.3f}°)")

    args.out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict()}
    torch.save(state, args.out_checkpoint)
    print(f"[finetune] ✓ saved {args.out_checkpoint}")
    print(f"[finetune] next: python3 convert_eth_xgaze_to_coreml.py "
          f"--ckpt {args.out_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
