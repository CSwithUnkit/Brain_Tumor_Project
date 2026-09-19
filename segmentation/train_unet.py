import argparse
import os
import sys

# Force UTF-8 encoding for progress bar characters on Windows terminals
if hasattr(sys.stdout, 'reconfigure') and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import json
import logging
import random
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision.transforms as T
from data.dataset_preprocessor import BRISCSegmentationDataset

from segmentation.unet_model import UNet
from segmentation.metrics import (
    CombinedBCEDiceLoss, TverskyFocalLoss,
    compute_dice_coefficient, compute_iou_score,
    compute_precision_recall, compute_hausdorff_distance
)
from utils.device_config import get_system_execution_profile, atomic_torch_save, atomic_json_save

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_type', type=str, choices=['original', 'enhanced'], default='original')
    parser.add_argument('--epochs', type=int, default=25)
    # Default None → fall back to dynamic hardware profile when not specified
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size. If omitted, auto-detected from hardware profile.')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader workers. If omitted, auto-detected from hardware profile.')
    parser.add_argument('--lr', type=float, default=1e-4)
    return parser.parse_args()

def seed_everything(seed: int = 42) -> None:
    """Enforce full reproducibility across all backends (FR-011, NFR-005)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = get_args()
    seed_everything(42)

    # ── Dynamic hardware profile ──────────────────────────────────────────────
    # Detects GPU/CPU, VRAM, RAM, and derives safe batch_size / num_workers.
    # CLI args override profile values when explicitly supplied by the user.
    profile = get_system_execution_profile()
    device      = profile["device"]
    batch_size  = args.batch_size  if args.batch_size  is not None else profile["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else profile["num_workers"]
    use_amp     = profile["use_amp"]
    pin_memory  = profile["pin_memory"]
    persistent_workers = profile["persistent_workers"]

    logger.info(
        f"ℹ️ [INFO] Hardware Profile → Device: {device} ({profile['gpu_name']}) | "
        f"VRAM: {profile['vram_gb']} GB | RAM: {profile['total_ram_gb']} GB | "
        f"Batch: {batch_size} | Workers: {num_workers} | AMP: {use_amp}"
    )
    logger.info(f"ℹ️ [INFO] Using device: {device}")
    
    os.makedirs('checkpoints/unet', exist_ok=True)
    tb_dir = f"runs/unet/unet_{args.input_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(tb_dir)
    
    model = UNet(n_channels=3, n_classes=1).to(device)
    pos_weight = torch.tensor([10.0]).to(device)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    tversky_criterion = TverskyFocalLoss(alpha=0.7, beta=0.3, gamma=0.75)
    criterion = lambda logits, targets: (
        0.4 * bce_criterion(logits, targets) + 0.6 * tversky_criterion(logits, targets)
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-7
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    best_dice = 0.0
    patience = 7
    epochs_no_improve = 0
    history = []
    
    meta_path = 'data/brisc/brisc_metadata.json'
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"BRISC metadata not found at {meta_path}. Please run: python -m data.dataset_ingestion")
    
    with open(meta_path, 'r') as f:
        brisc_data = json.load(f)
        
    seg_pairs = brisc_data.get('segmentation', [])
    if len(seg_pairs) == 0:
        raise FileNotFoundError("0 segmentation pairs found. Please re-run python -m data.dataset_ingestion")

    logger.info(f"ℹ️ [INFO] Loaded {len(seg_pairs)} segmentation pairs for training.")
    random.shuffle(seg_pairs)
    split_idx = int(len(seg_pairs) * 0.7)
    train_pairs = seg_pairs[:split_idx]
    val_pairs = seg_pairs[split_idx:]
    
    # Use BRISCSegmentationDataset with proper Albumentations + ImageNet normalization
    train_dataset = BRISCSegmentationDataset(train_pairs, split='train')
    val_dataset   = BRISCSegmentationDataset(val_pairs, split='val')

    # DataLoader: use hardware-profile-derived values for safe cross-platform operation.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=pin_memory, num_workers=num_workers,
        persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=pin_memory, num_workers=num_workers,
        persistent_workers=persistent_workers
    )
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t_epoch_start = time.time()

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            # AMP forward pass: use_amp=True on CUDA (fp16), False on CPU
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

            # In-place single-line progress: update every 25 batches
            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(train_loader):
                progress = int(30 * (batch_idx + 1) / len(train_loader))
                bar = "█" * progress + "░" * (30 - progress)
                sys.stdout.write(
                    f"\rEpoch {epoch:02d}/{args.epochs:02d} [{bar}] "
                    f"{batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}"
                )
                sys.stdout.flush()

        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_dice, val_loss = 0.0, 0.0
        val_iou, val_hd95 = 0.0, 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, masks)
                    
                val_loss += loss.item()
                val_dice += compute_dice_coefficient(logits, masks)
                val_iou += compute_iou_score(logits, masks)
                val_hd95 += compute_hausdorff_distance(logits, masks)
                
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        val_hd95 /= len(val_loader)
        
        scheduler.step(val_dice)

        elapsed = time.time() - t_epoch_start
        # End-of-epoch: overwrite the progress bar with the finalized summary
        sys.stdout.write(
            f"\rEpoch {epoch:02d}/{args.epochs:02d} — "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | "
            f"HD95: {val_hd95:.2f}px | Time: {elapsed:.1f}s\n"
        )
        sys.stdout.flush()

        logger.info(f"ℹ️ [INFO] Epoch {epoch}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val Dice {val_dice:.4f}")
        logger.info(f"  ℹ️ [INFO] IoU: {val_iou:.4f} | HD95: {val_hd95:.2f}px")
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Metric/Dice', val_dice, epoch)
        writer.add_scalar('Metric/IoU', val_iou, epoch)
        writer.add_scalar('Metric/HD95', val_hd95, epoch)
        
        if val_dice > best_dice:
            best_dice = val_dice
            epochs_no_improve = 0
            # Atomic save: prevents checkpoint corruption on Google Drive sync folders
            ckpt_name = f"checkpoints/unet/best_unet_{args.input_type}.pth"
            atomic_torch_save(model.state_dict(), ckpt_name)
            logger.info(f"  ✅ [SUCCESS] New best Dice={val_dice:.4f} checkpoint saved (atomic): {ckpt_name}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"⚠️ [WARN] Early stopping triggered at epoch {epoch}. Best Dice: {best_dice:.4f}")
                break
            
        history.append({'epoch': epoch, 'val_dice': val_dice})
        
        # FR: Memory Cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Atomic JSON save: prevents partial-write corruption on synced drives
    atomic_json_save(history, f"checkpoints/unet/unet_training_history_{args.input_type}.json")
        
    writer.close()

if __name__ == '__main__':
    main()
