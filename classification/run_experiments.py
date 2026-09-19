import argparse
import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from torch.utils.data import DataLoader
from classification.classifier_model import BrainTumorClassifier
from classification.masking_utils import apply_hard_mask
from data.dataset_preprocessor import BRISCClassificationDataset
from utils.device_config import get_system_execution_profile, atomic_torch_save, atomic_json_save

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['exp1_baseline', 'exp2_enhanced', 'exp3_seg_guided', 'exp3_soft_masked'],
                        help='Experiment to run.')
    parser.add_argument('--epochs', type=int, default=25)
    # Default None → fall back to dynamic hardware profile when not specified
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size. If omitted, auto-detected from hardware profile.')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader workers. If omitted, auto-detected from hardware profile.')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--unet_checkpoint', type=str, default=None,
                        help='Path to trained U-Net for Exp 3 variants')
    parser.add_argument('--in_notebook', action='store_true',
                        help='Enable IPython-compatible in-place progress (clear_output per epoch).')
    return parser.parse_args()

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute comprehensive multiclass metrics."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()
    
    return {
        'accuracy': acc,
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'confusion_matrix': conf_matrix
    }

def seed_everything(seed: int = 42) -> None:
    """Enforce full reproducibility across all backends (FR-011, NFR-005)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
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
    logger.info(f"ℹ️ [INFO] Running {args.experiment} on device: {device}")
    
    os.makedirs('checkpoints/classification', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    tb_dir = f"runs/{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(tb_dir)
    
    model = BrainTumorClassifier(num_classes=4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # === STAGE 1: Head Warmup (epochs 1 to WARMUP_EPOCHS) ===
    WARMUP_EPOCHS = 5
    model.freeze_feature_extractor(freeze=True)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # Load real BRISC classification data from metadata
    import json
    meta_path = 'data/brisc/brisc_metadata.json'
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"BRISC metadata not found at {meta_path}. Please run: python -m data.dataset_ingestion")
    
    with open(meta_path, 'r') as f:
        brisc_meta = json.load(f)
    
    all_records = brisc_meta.get('classification', [])
    if len(all_records) == 0:
        raise FileNotFoundError("0 classification images found. Please re-run python -m data.dataset_ingestion")
    
    # If Exp 2 or Exp 3, prefer enhanced cached images.
    # Match by basename; warn (but keep original path) if cache entry is missing.
    if args.experiment in ['exp2_enhanced', 'exp3_seg_guided']:
        cache_dir = 'data/cached_enhanced'
        missing_cache = 0
        for rec in all_records:
            cached = os.path.join(cache_dir, os.path.basename(rec['path']))
            if os.path.exists(cached):
                rec['path'] = cached
            else:
                missing_cache += 1
        if missing_cache > 0:
            logger.warning(
                f"⚠️ [WARN] {missing_cache} image(s) not found in {cache_dir}; "
                "falling back to raw image paths for those entries."
            )
    
    random.shuffle(all_records)
    class_to_idx = {'glioma': 0, 'meningioma': 1, 'pituitary': 2, 'no_tumor': 3}
    split_idx_train = int(len(all_records) * 0.70)
    split_idx_val   = int(len(all_records) * 0.85)
    train_records = all_records[:split_idx_train]
    val_records   = all_records[split_idx_train:split_idx_val]
    
    train_dataset = BRISCClassificationDataset(train_records, split='train', class_to_idx=class_to_idx)
    val_dataset   = BRISCClassificationDataset(val_records,   split='val',   class_to_idx=class_to_idx)

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
    
    logger.info(f"ℹ️ [INFO] Train: {len(train_dataset)} | Val: {len(val_dataset)} samples")
    
    best_val_f1 = 0.0
    history = []

    # ── Exp 3: load U-Net for on-the-fly segmentation-guided masking ──────────
    unet_model: Optional[torch.nn.Module] = None
    if args.experiment == 'exp3_seg_guided':
        unet_ckpt = args.unet_checkpoint or 'checkpoints/unet/best_unet_enhanced.pth'
        if os.path.exists(unet_ckpt):
            from segmentation.unet_model import UNet
            unet_model = UNet(n_channels=3, n_classes=1).to(device)
            unet_model.load_state_dict(
                torch.load(unet_ckpt, map_location=device, weights_only=True)
            )
            unet_model.eval()
            logger.info(f"ℹ️ [INFO] Exp3: U-Net loaded from {unet_ckpt}")
        else:
            logger.warning(
                f"⚠️ [WARN] Exp3: U-Net checkpoint not found at {unet_ckpt}. "
                "Training will proceed without segmentation masking."
            )
    
    for epoch in range(1, args.epochs + 1):
        if epoch == WARMUP_EPOCHS + 1:
            logger.info("ℹ️ [INFO] ==> Stage 2: Unfreezing top backbone blocks for differential fine-tuning")
            model.freeze_feature_extractor(freeze=False)
            optimizer = optim.AdamW([
                {'params': [p for n, p in model.model.named_parameters()
                            if 'classifier' not in n], 'lr': 1e-5},
                {'params': model.model.classifier.parameters(), 'lr': 1e-4},
            ], weight_decay=1e-4)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.epochs - WARMUP_EPOCHS, eta_min=1e-6
            )
            
        # Training
        model.train()
        train_loss = 0.0
        t_epoch_start = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            # ── Exp 3: generate U-Net masks and apply hard-mask RoI crop ──────
            if unet_model is not None:
                with torch.no_grad():
                    mask_logits = unet_model(images)          # (B, 1, H, W)
                    mask_prob   = torch.sigmoid(mask_logits)  # probabilities

                # Guard: skip masking for samples with empty masks (< 50 pixels)
                binary_mask = (mask_prob > 0.5).float()
                mask_areas  = binary_mask.view(binary_mask.size(0), -1).sum(dim=1)  # (B,)
                valid_mask  = (mask_areas >= 50).float().view(-1, 1, 1, 1)

                # apply_hard_mask handles per-sample fallback internally, but we
                # additionally zero-out mask_prob where area < 50 so that
                # apply_hard_mask receives an explicitly empty mask and falls back.
                safe_mask_prob = mask_prob * valid_mask
                images = apply_hard_mask(images, safe_mask_prob, padding=0.15)

            # AMP forward pass: use_amp=True on CUDA (fp16), False on CPU
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            # In-place single-line progress: update every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                batches_done  = batch_idx + 1
                elapsed_so_far = time.time() - t_epoch_start
                eta = elapsed_so_far / batches_done * (len(train_loader) - batches_done)
                progress = int(30 * batches_done / len(train_loader))
                bar = "█" * progress + "░" * (30 - progress)
                sys.stdout.write(
                    f"\rEpoch {epoch:02d}/{args.epochs:02d} [{bar}] "
                    f"Batch {batches_done:03d}/{len(train_loader):03d} "
                    f"- Loss: {loss.item():.4f} - ETA: {eta:.0f}s"
                )
                sys.stdout.flush()

        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, labels)
                    
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        val_acc = metrics['accuracy']
        val_f1 = metrics['macro_f1']

        elapsed = time.time() - t_epoch_start
        # End-of-epoch: overwrite in-progress bar with the finalized summary
        sys.stdout.write(
            f"\rEpoch {epoch:02d}/{args.epochs:02d} — "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | Macro F1: {val_f1*100:.2f}% | Time: {elapsed:.1f}s\n"
        )
        sys.stdout.flush()

        logger.info(f"ℹ️ [INFO] Epoch {epoch}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f} | Val F1 {val_f1:.4f}")
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Metric/Accuracy', val_acc, epoch)
        writer.add_scalar('Metric/MacroF1', val_f1, epoch)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Atomic save: prevents checkpoint corruption on Google Drive sync folders
            ckpt_path = f"checkpoints/classification/best_efficientnet_{args.experiment}.pth"
            atomic_torch_save(model.state_dict(), ckpt_path)
            logger.info(f"  ✅ [SUCCESS] New best F1={val_f1:.4f} checkpoint saved (atomic)")
            
        history.append({'epoch': epoch, 'val_metrics': metrics})
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Atomic JSON save: prevents partial-write corruption on synced drives
    atomic_json_save(history, f"results/metrics_{args.experiment}.json")
        
    writer.close()

if __name__ == '__main__':
    main()
