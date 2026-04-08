#!/usr/bin/env python3
"""
Self-Supervised Learning (SimCLR) trainer for plankton crops.

Trains a ResNet-50 backbone with the SimCLR contrastive learning framework
(Chen et al. 2020: https://arxiv.org/abs/2002.05709).

Designed to learn fine-grained morphological features within a specific
category (e.g. non-living particles) so that downstream UMAP+HDBSCAN
clustering produces meaningful sub-groups.

Checkpoint format
-----------------
{
    "backbone_state_dict": ...,   # ResNet-50 sans final FC
    "projector_state_dict": ...,  # 2-layer MLP projection head
    "epoch": int,
    "loss": float,
    "input_size": int,
    "feature_dim": int,           # 2048
}
Only the backbone is needed at inference time.
"""

import os
import random
import logging
import argparse
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Augmentation  (plankton-appropriate: minimal colour, strong spatial)
# ---------------------------------------------------------------------------

class GaussianBlur:
    def __init__(self, sigma_range=(0.1, 2.0)):
        self.sigma_range = sigma_range

    def __call__(self, img):
        sigma = random.uniform(*self.sigma_range)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def build_augmentation(input_size=224, profile: str = "default"):
    profile = str(profile).strip().lower()
    if profile in {"size_aware", "size_aware_grayscale", "grayscale_size_aware"}:
        return T.Compose([
            T.Resize(input_size + 32),
            T.CenterCrop(input_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=8, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.25),
            T.RandomGrayscale(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if profile in {"sparse", "sparse_grayscale", "grayscale_sparse"}:
        return T.Compose([
            T.RandomResizedCrop(
                size=input_size,
                scale=(0.45, 1.0),
                ratio=(0.85, 1.15),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=12, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.35),
            T.RandomGrayscale(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return T.Compose([
        T.RandomResizedCrop(
            size=input_size,
            scale=(0.15, 1.0),
            ratio=(0.75, 1.33),
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply(
            [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.05, hue=0.0)],
            p=0.8,
        ),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([GaussianBlur(sigma_range=(0.1, 2.0))], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_inference_transform(input_size=224):
    return T.Compose([
        T.Resize(input_size + 32),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CropDataset(Dataset):
    def __init__(
        self,
        image_paths,
        augment=None,
        input_size=224,
        inference=False,
        validate_paths=True,
        foreground_crop=False,
        foreground_threshold=245,
        foreground_min_pixels=8,
        mask_dir: Optional[str] = None,
        mask_suffix: str = "",
        mask_extension: str = ".png",
        size_aware_canvas: bool = False,
        canvas_size: int = 512,
    ):
        if validate_paths:
            self.paths = [p for p in image_paths if os.path.isfile(p)]
            skipped = len(image_paths) - len(self.paths)
            if skipped:
                log.warning(f"{skipped} image paths skipped (not found)")
        else:
            # Faster startup for very large datasets: skip millions of os.path.isfile calls.
            # Missing/corrupt images are handled in _load() with a zero-image fallback.
            self.paths = list(image_paths)
        self.augment = augment or build_augmentation(input_size)
        self.inference_transform = build_inference_transform(input_size)
        self.inference = inference
        self.foreground_crop = bool(foreground_crop) and (not inference)
        self.foreground_threshold = int(foreground_threshold)
        self.foreground_min_pixels = int(foreground_min_pixels)
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.mask_extension = mask_extension
        self.size_aware_canvas = bool(size_aware_canvas)
        self.canvas_size = int(canvas_size)

    def __len__(self):
        return len(self.paths)

    def _load(self, path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))

    def _resolve_mask_path(self, image_path: str) -> Optional[str]:
        if not self.mask_dir:
            return None
        base = os.path.splitext(os.path.basename(image_path))[0]
        candidate = os.path.join(
            self.mask_dir,
            f"{base}{self.mask_suffix}{self.mask_extension}",
        )
        return candidate if os.path.isfile(candidate) else None

    def _foreground_bbox(self, image_path: str, img: Image.Image) -> Optional[tuple[int, int, int, int]]:
        mask_arr = None
        mask_path = self._resolve_mask_path(image_path)
        if mask_path is not None:
            try:
                mask_arr = np.array(Image.open(mask_path).convert("L")) > 0
            except Exception:
                mask_arr = None

        if mask_arr is None:
            gray = np.array(img.convert("L"))
            mask_arr = gray < self.foreground_threshold

        ys, xs = np.where(mask_arr)
        if len(xs) < self.foreground_min_pixels:
            return None

        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        w, h = img.size
        pad = max(2, int(0.1 * max(x1 - x0 + 1, y1 - y0 + 1)))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w - 1, x1 + pad)
        y1 = min(h - 1, y1 + pad)
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1 + 1, y1 + 1)

    def _crop_to_foreground(self, image_path: str, img: Image.Image) -> Image.Image:
        bbox = self._foreground_bbox(image_path, img)
        if bbox is None:
            return img
        return img.crop(bbox)

    def _to_size_canvas(self, img: Image.Image) -> Image.Image:
        if not self.size_aware_canvas:
            return img
        side = max(self.canvas_size, max(img.size))
        canvas = Image.new("RGB", (side, side), color=(255, 255, 255))
        x = (side - img.size[0]) // 2
        y = (side - img.size[1]) // 2
        canvas.paste(img, (x, y))
        return canvas

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        img = self._load(image_path)
        if self.foreground_crop:
            img = self._crop_to_foreground(image_path, img)
        img = self._to_size_canvas(img)
        if self.inference:
            return self.inference_transform(img), image_path
        return self.augment(img), self.augment(img)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SimCLRNet(nn.Module):
    """ResNet-50 backbone + 2-layer MLP projection head (SimCLR)."""

    def __init__(self, pretrained=True, projection_dim=128, hidden_dim=512):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.feature_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        h = self.backbone(x)   # (B, 2048)
        z = self.projector(h)  # (B, projection_dim)
        return h, z

    def get_features(self, x):
        """L2-normalised backbone features, no gradient."""
        with torch.no_grad():
            h = self.backbone(x)
        return F.normalize(h, dim=1)


# ---------------------------------------------------------------------------
# NT-Xent loss  (SimCLR Eq. 1)
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.xe = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))
        labels = torch.cat(
            [torch.arange(B, 2 * B), torch.arange(0, B)], dim=0
        ).to(z.device)
        return self.xe(sim, labels)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _save_checkpoint(model, epoch, loss, input_size, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "backbone_state_dict": model.backbone.state_dict(),
        "projector_state_dict": model.projector.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "input_size": input_size,
        "feature_dim": model.feature_dim,
    }, path)


def train_ssl_model(
    image_paths: List[str],
    output_path: str,
    pretrained_backbone: bool = True,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.07,
    projection_dim: int = 128,
    input_size: int = 224,
    num_workers: int = 4,
    save_every: int = 50,
    log_every_batches: int = 200,
    device: Optional[str] = None,
    resume_path: Optional[str] = None,
    augmentation_profile: str = "default",
    foreground_crop: bool = False,
    foreground_threshold: int = 245,
    foreground_min_pixels: int = 8,
    mask_dir: Optional[str] = None,
    mask_suffix: str = "",
    mask_extension: str = ".png",
    positive_consistency_weight: float = 0.0,
    size_aware_canvas: bool = False,
    canvas_size: int = 512,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> str:
    """
    Train SimCLR on *image_paths* and save checkpoint to *output_path*.
    Returns the output_path string.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    log.info(f"Device: {dev}  |  images: {len(image_paths)}  |  epochs: {epochs}")

    dataset = CropDataset(
        image_paths,
        augment=build_augmentation(input_size=input_size, profile=augmentation_profile),
        input_size=input_size,
        validate_paths=False,
        foreground_crop=foreground_crop,
        foreground_threshold=foreground_threshold,
        foreground_min_pixels=foreground_min_pixels,
        mask_dir=mask_dir,
        mask_suffix=mask_suffix,
        mask_extension=mask_extension,
        size_aware_canvas=size_aware_canvas,
        canvas_size=canvas_size,
    )
    if len(dataset) == 0:
        raise RuntimeError("No valid images found for SSL training.")

    effective_batch = min(batch_size, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=effective_batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    log.info(f"Training loader ready: batches/epoch={len(loader):,}, batch_size={effective_batch}")
    log.info(
        "SSL training options: "
        f"aug_profile={augmentation_profile}, "
        f"foreground_crop={foreground_crop}, "
        f"size_aware_canvas={size_aware_canvas}, "
        f"canvas_size={canvas_size}, "
        f"positive_consistency_weight={positive_consistency_weight}"
    )

    model = SimCLRNet(
        pretrained=pretrained_backbone,
        projection_dim=projection_dim,
    ).to(dev)

    start_epoch = 0
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=dev)
        model.backbone.load_state_dict(ckpt["backbone_state_dict"])
        model.projector.load_state_dict(ckpt["projector_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        log.info(f"Resumed from {resume_path}  (epoch {start_epoch})")

    criterion = NTXentLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    for _ in range(start_epoch):
        scheduler.step()

    best_loss = float("inf")
    for epoch in range(start_epoch + 1, epochs + 1):
        if stop_requested is not None and stop_requested():
            raise InterruptedError("Stopped by user during SSL training.")
        model.train()
        running = 0.0
        running_pos = 0.0
        epoch_start = time.time()
        total_batches = len(loader)
        for batch_idx, (v1, v2) in enumerate(loader, 1):
            if stop_requested is not None and stop_requested():
                raise InterruptedError("Stopped by user during SSL training.")
            v1, v2 = v1.to(dev, non_blocking=True), v2.to(dev, non_blocking=True)
            optimizer.zero_grad()
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss_contrastive = criterion(z1, z2)
            if positive_consistency_weight > 0:
                z1n = F.normalize(z1, dim=1)
                z2n = F.normalize(z2, dim=1)
                loss_positive = (1.0 - F.cosine_similarity(z1n, z2n, dim=1)).mean()
                loss = loss_contrastive + positive_consistency_weight * loss_positive
                running_pos += float(loss_positive.item())
            else:
                loss = loss_contrastive
            loss.backward()
            optimizer.step()
            running += loss.item()

            if (
                log_every_batches > 0
                and (batch_idx % log_every_batches == 0 or batch_idx == total_batches)
            ):
                elapsed = time.time() - epoch_start
                batches_done = batch_idx
                sec_per_batch = elapsed / max(batches_done, 1)
                remaining_batches = total_batches - batches_done
                eta_sec = sec_per_batch * remaining_batches
                eta_min = eta_sec / 60.0
                avg_so_far = running / max(batches_done, 1)
                pct = 100.0 * batches_done / max(total_batches, 1)
                pos_txt = ""
                if positive_consistency_weight > 0:
                    pos_txt = f" | avg_pos={running_pos / max(batches_done, 1):.4f}"
                log.info(
                    f"    Epoch {epoch}/{epochs} | "
                    f"batch {batches_done}/{total_batches} ({pct:.1f}%) | "
                    f"avg_loss={avg_so_far:.4f} | "
                    f"ETA~{eta_min:.1f} min"
                    f"{pos_txt}"
                )
        scheduler.step()
        avg = running / len(loader)
        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"  Epoch {epoch:4d}/{epochs}  loss={avg:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
        if epoch % save_every == 0:
            _save_checkpoint(
                model, epoch, avg, input_size,
                output_path.replace(".pth", f"_ep{epoch:04d}.pth"),
            )
        if avg < best_loss:
            best_loss = avg

    _save_checkpoint(model, epochs, best_loss, input_size, output_path)
    log.info(f"Training complete — best loss={best_loss:.4f}  →  {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Feature extraction (inference)
# ---------------------------------------------------------------------------

def load_backbone(
    checkpoint_path: str,
    device: Optional[str] = None,
) -> Tuple[SimCLRNet, int]:
    """Load trained backbone. Returns (SimCLRNet in eval(), input_size)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=dev)
    input_size = ckpt.get("input_size", 224)
    model = SimCLRNet(pretrained=False)
    model.backbone.load_state_dict(ckpt["backbone_state_dict"])
    if "projector_state_dict" in ckpt:
        model.projector.load_state_dict(ckpt["projector_state_dict"])
    model = model.to(dev)
    model.eval()
    log.info(
        f"Loaded backbone from {checkpoint_path} "
        f"(epoch={ckpt.get('epoch', '?')}, loss={ckpt.get('loss', float('nan')):.4f})"
    )
    return model, input_size


def extract_features(
    image_paths: List[str],
    checkpoint_path: str,
    batch_size: int = 512,
    num_workers: int = 4,
    device: Optional[str] = None,
    input_size: int = 224,
    log_every_batches: int = 200,
    size_aware_canvas: bool = False,
    canvas_size: int = 512,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract L2-normalised 2048-d backbone features.

    Returns
    -------
    features    : np.ndarray shape (N, 2048)
    valid_paths : list[str]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model, ckpt_size = load_backbone(checkpoint_path, device)
    actual_size = ckpt_size or input_size

    dataset = CropDataset(
        image_paths,
        input_size=actual_size,
        inference=True,
        size_aware_canvas=size_aware_canvas,
        canvas_size=canvas_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    all_feats: List[np.ndarray] = []
    all_paths: List[str] = []
    log.info(f"Extracting features from {len(dataset)} images ...")
    total_batches = len(loader)
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (imgs, paths) in enumerate(loader, 1):
            if stop_requested is not None and stop_requested():
                raise InterruptedError("Stopped by user during feature extraction.")
            imgs = imgs.to(dev, non_blocking=True)
            feats = model.get_features(imgs)
            all_feats.append(feats.cpu().numpy())
            all_paths.extend(list(paths))

            if (
                log_every_batches > 0
                and (batch_idx % log_every_batches == 0 or batch_idx == total_batches)
            ):
                elapsed = time.time() - t0
                sec_per_batch = elapsed / max(batch_idx, 1)
                remaining = total_batches - batch_idx
                eta_min = (sec_per_batch * remaining) / 60.0
                pct = 100.0 * batch_idx / max(total_batches, 1)
                log.info(
                    f"    Extract features | batch {batch_idx}/{total_batches} ({pct:.1f}%) | "
                    f"ETA~{eta_min:.1f} min"
                )

    features = np.concatenate(all_feats, axis=0)
    log.info(f"Features shape: {features.shape}")
    return features, all_paths


def extract_features_imagenet(
    image_paths: List[str],
    batch_size: int = 512,
    num_workers: int = 4,
    device: Optional[str] = None,
    input_size: int = 224,
    log_every_batches: int = 200,
    size_aware_canvas: bool = False,
    canvas_size: int = 512,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract L2-normalised ResNet-50 ImageNet features (no SSL checkpoint).

    Returns
    -------
    features    : np.ndarray shape (N, 2048)
    valid_paths : list[str]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    weights = models.ResNet50_Weights.IMAGENET1K_V1
    backbone = models.resnet50(weights=weights)
    backbone.fc = nn.Identity()
    backbone = backbone.to(dev)
    backbone.eval()

    dataset = CropDataset(
        image_paths,
        input_size=input_size,
        inference=True,
        size_aware_canvas=size_aware_canvas,
        canvas_size=canvas_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    all_feats: List[np.ndarray] = []
    all_paths: List[str] = []
    log.info(f"Extracting ImageNet features from {len(dataset)} images ...")
    total_batches = len(loader)
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (imgs, paths) in enumerate(loader, 1):
            if stop_requested is not None and stop_requested():
                raise InterruptedError("Stopped by user during ImageNet feature extraction.")
            imgs = imgs.to(dev, non_blocking=True)
            feats = backbone(imgs)
            feats = F.normalize(feats, dim=1)
            all_feats.append(feats.cpu().numpy())
            all_paths.extend(list(paths))

            if (
                log_every_batches > 0
                and (batch_idx % log_every_batches == 0 or batch_idx == total_batches)
            ):
                elapsed = time.time() - t0
                sec_per_batch = elapsed / max(batch_idx, 1)
                remaining = total_batches - batch_idx
                eta_min = (sec_per_batch * remaining) / 60.0
                pct = 100.0 * batch_idx / max(total_batches, 1)
                log.info(
                    f"    Extract ImageNet features | batch {batch_idx}/{total_batches} ({pct:.1f}%) | "
                    f"ETA~{eta_min:.1f} min"
                )

    features = np.concatenate(all_feats, axis=0)
    log.info(f"ImageNet feature extraction complete: {features.shape}")
    return features, all_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SimCLR on plankton crops",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image-paths", required=True,
                   help="Text file: one absolute image path per line")
    p.add_argument("--output", required=True, help="Output .pth path")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every-batches", type=int, default=200,
                   help="Log training progress every N batches (0 disables batch logs)")
    p.add_argument("--aug-profile", choices=["default", "sparse_grayscale"], default="default",
                   help="Augmentation profile")
    p.add_argument("--foreground-crop", action="store_true",
                   help="Crop around foreground before augmentations")
    p.add_argument("--foreground-threshold", type=int, default=245,
                   help="Auto foreground threshold on grayscale intensity")
    p.add_argument("--foreground-min-pixels", type=int, default=8,
                   help="Minimum foreground pixels required to crop")
    p.add_argument("--mask-dir", default=None,
                   help="Optional external mask directory")
    p.add_argument("--mask-suffix", default="",
                   help="Suffix inserted before mask extension (e.g. _mask)")
    p.add_argument("--mask-extension", default=".png",
                   help="Mask extension (default: .png)")
    p.add_argument("--positive-consistency-weight", type=float, default=0.0,
                   help="Extra positive cosine consistency weight added to NT-Xent")
    p.add_argument("--device", default=None, help="cuda | cpu")
    p.add_argument("--resume", default=None, help="Resume from checkpoint")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip ImageNet initialisation")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.image_paths) as fh:
        paths = [line.strip() for line in fh if line.strip()]
    train_ssl_model(
        image_paths=paths,
        output_path=args.output,
        pretrained_backbone=not args.no_pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        projection_dim=args.projection_dim,
        input_size=args.input_size,
        num_workers=args.num_workers,
        log_every_batches=args.log_every_batches,
        augmentation_profile=args.aug_profile,
        foreground_crop=args.foreground_crop,
        foreground_threshold=args.foreground_threshold,
        foreground_min_pixels=args.foreground_min_pixels,
        mask_dir=args.mask_dir,
        mask_suffix=args.mask_suffix,
        mask_extension=args.mask_extension,
        positive_consistency_weight=args.positive_consistency_weight,
        device=args.device,
        resume_path=args.resume,
    )
