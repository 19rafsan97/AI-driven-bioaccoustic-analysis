import argparse
import os
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import ASTForAudioClassification
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.optim import AdamW


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_2d_mel(arr: np.ndarray) -> np.ndarray:
    """Ensure array is float32 with shape (mel_bins=128, time). Transpose if needed."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mel-spectrogram, got shape {arr.shape}")
    h, w = arr.shape
    # If the first dim looks like time (typically > 128) and second is 128, transpose
    if h != 128 and w == 128:
        arr = arr.T
    elif h == 128:
        pass
    else:
        # Last resort: if neither dim is 128, keep as is but warn
        if 128 not in (h, w):
            # Auto-resize by pad/crop frequency to 128
            if h < 128:
                pad = 128 - h
                arr = np.pad(arr, ((0, pad), (0, 0)), mode='edge')
            elif h > 128:
                arr = arr[:128, :]
    return arr.astype(np.float32, copy=False)


def compute_mean_std(paths, sample_frac=1.0):
    """Compute dataset mean/std over mel bins and time from given .npy paths.
    Uses streaming mean/var to avoid memory spikes."""
    n = 0
    mean = 0.0
    M2 = 0.0
    total_elems = 0

    use_paths = paths
    if 0 < sample_frac < 1.0:
        k = max(1, int(len(paths) * sample_frac))
        use_paths = random.sample(paths, k)

    for p in use_paths:
        x = np.load(p)
        x = ensure_2d_mel(x)
        x = x.reshape(-1).astype(np.float32)
        m = x.mean()
        s2 = x.var()
        count = x.size
        total_elems += count
        if n == 0:
            mean = m
            M2 = s2 * count
            n = count
        else:
            # Parallel variance
            delta = m - mean
            new_count = n + count
            mean += delta * (count / new_count)
            M2 += s2 * count + delta**2 * (n * count / new_count)
            n = new_count
    var = M2 / max(total_elems, 1)
    std = math.sqrt(max(var, 1e-12))
    return float(mean), float(std)


def spec_augment(mel: np.ndarray, max_time_mask_frac=0.15, max_freq_mask_frac=0.15,
                 num_time_masks=2, num_freq_masks=2, replace_with_zero=True) -> np.ndarray:
    """Simple SpecAugment (time/frequency masking). mel shape: (128, T)."""
    mel = mel.copy()
    n_mels, T = mel.shape

    for _ in range(num_freq_masks):
        f = random.randint(0, max(1, int(n_mels * max_freq_mask_frac)))
        f0 = random.randint(0, max(0, n_mels - f))
        if replace_with_zero:
            mel[f0:f0+f, :] = 0.0
        else:
            mel[f0:f0+f, :] = mel.mean()

    for _ in range(num_time_masks):
        t = random.randint(0, max(1, int(T * max_time_mask_frac)))
        t0 = random.randint(0, max(0, T - t))
        if replace_with_zero:
            mel[:, t0:t0+t] = 0.0
        else:
            mel[:, t0:t0+t] = mel.mean()

    return mel


class MixupHelper:
    def __init__(self, alpha=0.2, enabled=False):
        self.alpha = alpha
        self.enabled = enabled and alpha > 0
        self.beta = torch.distributions.beta.Beta(alpha, alpha) if self.enabled else None

    def apply(self, x, y):
        if not self.enabled:
            return x, y, None
        if x.size(0) < 2:
            return x, y, None
        # Shuffle indices
        perm = torch.randperm(x.size(0), device=x.device)
        x2, y2 = x[perm], y[perm]
        lam = self.beta.sample().to(x.device)
        x_mix = lam * x + (1 - lam) * x2
        # For CE with integer labels, we use mixed loss later: lam*CE(logits,y) + (1-lam)*CE(logits,y2)
        return x_mix, (y, y2), lam


# -----------------------------
# Dataset
# -----------------------------
class WMWBDataset(Dataset):
    def __init__(self, df: pd.DataFrame, spec_dir: Path, label2id: dict,
                 mean: float = 0.0, std: float = 1.0, augment=False,
                 target_frames: int | None = None,
                 specaugment_cfg: dict | None = None):
        self.df = df.reset_index(drop=True)
        self.spec_dir = Path(spec_dir)
        self.label2id = label2id
        self.mean = mean
        self.std = max(std, 1e-6)
        self.augment = augment
        self.target_frames = target_frames
        self.specaugment_cfg = specaugment_cfg or {}

    def __len__(self):
        return len(self.df)

    def _load_spec(self, row):
        # Construct path: either absolute path already or join with spec_dir
        spec_path = Path(row['spec_path'])
        if not spec_path.is_file():
            spec_path = self.spec_dir / row['spec_path']
        x = np.load(spec_path)
        x = ensure_2d_mel(x)
        return x

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def _pad_or_crop(self, x: np.ndarray) -> np.ndarray:
        if self.target_frames is None:
            return x
        n_mels, T = x.shape
        if T == self.target_frames:
            return x
        if T > self.target_frames:
            # center crop
            start = (T - self.target_frames) // 2
            return x[:, start:start+self.target_frames]
        else:
            # pad with edge
            pad = self.target_frames - T
            left = pad // 2
            right = pad - left
            return np.pad(x, ((0,0), (left, right)), mode='edge')

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = self._load_spec(row)
        if self.augment:
            x = spec_augment(x, **self.specaugment_cfg)
        x = self._normalize(x)
        x = self._pad_or_crop(x)
        # Convert to torch: (1, 128, T)
        x = torch.from_numpy(x)  # [128, T]  <-- no extra channel
        y = self.label2id[row['label']]
        y = torch.tensor(y, dtype=torch.long)
        return x, y


# -----------------------------
# Training / Evaluation
# -----------------------------

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targs = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            outputs = model(xb)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targs.append(yb.cpu().numpy())
    y_true = np.concatenate(all_targs)
    y_pred = np.concatenate(all_preds)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    return macro_f1, acc, y_true, y_pred


def save_confusion_matrix(y_true, y_pred, labels, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser(description="Fine-tune AST on WMWB spectrograms")

    # Data
    p.add_argument('--data_root', type=str, required=True, help='Root folder containing spectrograms.zip extracted and metadata.csv')
    p.add_argument('--spectrograms_dir', type=str, default='spectrograms', help='Relative or absolute path to spectrograms directory (species folders with .npy)')
    p.add_argument('--metadata_csv', type=str, default='metadata.csv', help='Path to metadata.csv (used for grouping by recording)')
    p.add_argument('--group_column', type=str, default='recording_id', help='Column name in metadata used for leakage-safe splits; if missing, auto-derive')
    p.add_argument('--label_column', type=str, default='species', help='Column with species label')
    p.add_argument('--spec_path_column', type=str, default='spec_filename', help='Column with .npy filename relative to spectrograms_dir; if missing, we glob')
    p.add_argument('--train_frac', type=float, default=0.8)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--test_frac', type=float, default=0.1)

    # Model / training
    p.add_argument('--model_name', type=str, default='MIT/ast-finetuned-audioset-10-10-0.4593')
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--grad_accum', type=int, default=2, help='Gradient accumulation steps')
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--weight_decay', type=float, default=5e-2)
    p.add_argument('--warmup_steps', type=int, default=500)
    p.add_argument('--label_smoothing', type=float, default=0.05)
    p.add_argument('--freeze_patch_embed_epochs', type=int, default=2, help='Warmup: freeze patch embed & pos embed for first N epochs')
    p.add_argument('--amp', action='store_true', help='Enable mixed precision training')
    p.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing')

    # Regularization
    p.add_argument('--use_class_weights', action='store_true', help='Use class weights in CrossEntropyLoss')
    p.add_argument('--specaug', action='store_true', help='Enable SpecAugment')
    p.add_argument('--mixup', type=float, default=0.0, help='Mixup alpha; 0 disables')

    # Normalization / framing
    p.add_argument('--compute_train_mean_std', action='store_true', help='Compute mean/std over TRAIN set (recommended)')
    p.add_argument('--target_frames', type=int, default=None, help='Force all specs to this time length via center crop/pad')

    # IO / logging
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--save_predictions', action='store_true')

    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_root = Path(args.data_root)
    spec_dir = Path(args.spectrograms_dir)
    if not spec_dir.is_absolute():
        spec_dir = data_root / spec_dir
    meta_path = Path(args.metadata_csv)
    if not meta_path.is_absolute():
        meta_path = data_root / meta_path

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'checkpoints').mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=str(out_dir / 'tb'))

    # ---------------------
    # Load metadata
    # ---------------------
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
    else:
        meta = pd.DataFrame()

    # If metadata missing columns, attempt to build a table by globbing the spectrograms directory
    if meta.empty or args.spec_path_column not in meta.columns or args.label_column not in meta.columns:
        rows = []
        for class_dir in sorted(spec_dir.glob('*')):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            for npy_path in class_dir.glob('*.npy'):
                # Heuristic for group/recording id: filename before first underscore
                stem = npy_path.stem
                group = stem.split('_')[0]
                rows.append({
                    'spec_filename': str(npy_path.relative_to(spec_dir)),
                    'species': label,
                    'recording_id': group
                })
        meta = pd.DataFrame(rows)

    # Normalize column names (in case of different casing)
    rename_map = {}
    for col in meta.columns:
        lc = col.lower()
        if lc == 'species' and col != 'species':
            rename_map[col] = 'species'
        if lc in ['recording', 'recording_id', 'xc_id', 'xeno_canto_id'] and col != 'recording_id':
            rename_map[col] = 'recording_id'
        if lc in ['spec', 'spectrogram', 'spec_filename', 'spectrogram_filename'] and col != 'spec_filename':
            rename_map[col] = 'spec_filename'
    if rename_map:
        meta = meta.rename(columns=rename_map)

    for req in [args.label_column, 'spec_filename']:
        if req not in meta.columns:
            raise ValueError(f"Required column '{req}' not found in metadata; available: {list(meta.columns)}")

    if args.group_column not in meta.columns:
        # Fall back to heuristic grouping
        meta[args.group_column] = meta['spec_filename'].apply(lambda s: Path(s).stem.split('_')[0])

    # Build full spec path
    meta['spec_path'] = meta['spec_filename'].apply(lambda s: str((spec_dir / s).resolve()))

    # Label maps
    classes = sorted(meta[args.label_column].unique())
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}

    # ---------------------
    # Split (leakage-safe by recording_id)
    # ---------------------
    gss = GroupShuffleSplit(n_splits=1, train_size=args.train_frac, random_state=args.seed)
    groups = meta[args.group_column].astype(str)
    y = meta[args.label_column].map(label2id).values
    idx_train, idx_temp = next(gss.split(meta, y, groups))
    temp = meta.iloc[idx_temp]

    # Split temp into val/test
    val_size = args.val_frac / max(args.val_frac + args.test_frac, 1e-8)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_size, random_state=args.seed+1)
    idx_val, idx_test = next(gss2.split(temp, temp[args.label_column].map(label2id).values, temp[args.group_column]))

    train_df = meta.iloc[idx_train].reset_index(drop=True)
    val_df = temp.iloc[idx_val].reset_index(drop=True)
    test_df = temp.iloc[idx_test].reset_index(drop=True)

    for df in (train_df, val_df, test_df):
        df['label'] = df[args.label_column].astype(str)  # args.label_column defaults to "species"
    # ---------------------
    # Normalization stats from TRAIN only (recommended)
    # ---------------------
    if args.compute_train_mean_std:
        train_paths = [Path(p) for p in train_df['spec_path'].tolist()]
        mean, std = compute_mean_std(train_paths, sample_frac=1.0)
    else:
        # Fallback: 0 mean / 1 std (assumes pre-normalized log-mels)
        mean, std = 0.0, 1.0

    # Determine target_frames from a sample if not provided
    if args.target_frames is None:
        sample_path = train_df['spec_path'].iloc[0]
        sample = ensure_2d_mel(np.load(sample_path))
        target_frames = sample.shape[1]
    else:
        target_frames = args.target_frames

    # Datasets
    specaug_cfg = dict(max_time_mask_frac=0.15, max_freq_mask_frac=0.15, num_time_masks=2, num_freq_masks=2)
    ds_train = WMWBDataset(train_df, spec_dir, label2id, mean, std, augment=args.specaug, target_frames=target_frames, specaugment_cfg=specaug_cfg)
    ds_val = WMWBDataset(val_df, spec_dir, label2id, mean, std, augment=False, target_frames=target_frames)
    ds_test = WMWBDataset(test_df, spec_dir, label2id, mean, std, augment=False, target_frames=target_frames)

    # Loaders
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # ---------------------
    # Model
    # ---------------------
    num_labels = len(classes)
    model = ASTForAudioClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    model.to(device)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Steps estimate for scheduler
    steps_per_epoch = max(1, len(train_loader) // max(1, args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    # Loss
    if args.use_class_weights:
        class_counts = train_df[args.label_column].map(label2id).value_counts().sort_index().values.astype(np.float32)
        weights = class_counts.sum() / (class_counts + 1e-6)
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        weights = None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    mixup = MixupHelper(alpha=args.mixup, enabled=(args.mixup > 0.0))

    # Optionally freeze patch embedding & positional embeddings for warmup epochs
    patch_embed_params = []
    pos_embed_params = []
    for n, p_ in model.named_parameters():
        if 'ast.patch_embed' in n or 'ast.pos_embed' in n:
            patch_embed_params.append((n, p_))
        if 'pos_embed' in n:
            pos_embed_params.append((n, p_))

    def set_patch_pos_requires_grad(req: bool):
        for _, p_ in patch_embed_params + pos_embed_params:
            p_.requires_grad = req

    best_f1 = -1.0
    best_path = out_dir / 'checkpoints' / 'best.pt'

    # Save label maps/config
    with open(out_dir / 'label2id.json', 'w') as f:
        json.dump(label2id, f, indent=2)
    with open(out_dir / 'id2label.json', 'w') as f:
        json.dump(id2label, f, indent=2)

    # ---------------------
    # Training loop
    # ---------------------
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        if epoch <= args.freeze_patch_embed_epochs:
            set_patch_pos_requires_grad(False)
        else:
            set_patch_pos_requires_grad(True)

        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            # Mixup
            if mixup.enabled:
                xb, (yb1, yb2), lam = mixup.apply(xb, yb)

            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(xb)
                logits = outputs.logits
                if mixup.enabled and lam is not None:
                    loss = lam * criterion(logits, yb1) + (1 - lam) * criterion(logits, yb2)
                else:
                    loss = criterion(logits, yb)
                loss = loss / max(1, args.grad_accum)

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running_loss += loss.item() * max(1, args.grad_accum)

        # Eval
        val_f1, val_acc, y_true_val, y_pred_val = evaluate(model, val_loader, device)
        writer.add_scalar('train/loss', running_loss / max(1, len(train_loader)), epoch)
        writer.add_scalar('val/macro_f1', val_f1, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)

        # Save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({'model_state_dict': model.state_dict(), 'label2id': label2id, 'id2label': id2label}, best_path)

        print(f"Epoch {epoch}: train_loss={running_loss/ max(1, len(train_loader)):.4f} val_f1={val_f1:.4f} val_acc={val_acc:.4f}")

    # ---------------------
    # Load best and Test
    # ---------------------
    if best_path.exists():
        chk = torch.load(best_path, map_location='cpu')
        model.load_state_dict(chk['model_state_dict'])

    test_f1, test_acc, y_true, y_pred = evaluate(model, test_loader, device)

    # Reports & artifacts
    report = classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(len(id2label))], digits=4)
    print("\nTest Results\n============\n" + report)

    with open(out_dir / 'classification_report.txt', 'w') as f:
        f.write(report)

    save_confusion_matrix(y_true, y_pred, [id2label[i] for i in range(len(id2label))], out_path=out_dir / 'confusion_matrix.png')

    metrics = {
        'best_val_macro_f1': best_f1,
        'test_macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'test_accuracy': float(accuracy_score(y_true, y_pred))
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    if args.save_predictions:
        pred_rows = []
        for idx, (t, p) in enumerate(zip(y_true, y_pred)):
            pred_rows.append({'true_id': int(t), 'true_label': id2label[int(t)], 'pred_id': int(p), 'pred_label': id2label[int(p)]})
        pd.DataFrame(pred_rows).to_csv(out_dir / 'predictions.csv', index=False)

    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
