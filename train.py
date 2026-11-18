"""
train.py — Dataset preparation, training, evaluation, and paper figure generation.

Usage
-----
  python train.py                   # train all runs (A, B, C) + generate figures
  python train.py --run cnn         # only PaperCNN (Run A: SGDM, Run B: Adam)
  python train.py --run baseline    # only pretrained baselines (Run C: DenseNet-121)
  python train.py --run figures     # regenerate figures from saved .pth weights (no training)
  python train.py --run figures --show  # also open interactive matplotlib windows
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from model import (
    CLASSES,
    CONFIG,
    FIGURES_DIR,
    IMAGENET_MEAN_DEV,
    IMAGENET_STD_DEV,
    IMG_EXTS,
    K,
    MODELS_DIR,
    PAPER_REF,
    WORK_DIR,
    PaperCNN,
    build_heatmap_overlay,
    build_pretrained,
    device,
    is_rgb_model,
    load_model,
    predict_with_gradcam,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================================================================
# 1. DATA LOADING & CACHING
# ===========================================================================

def locate_source() -> str:
    """Download/locate the COVID-19 Radiography Database via kagglehub."""
    import kagglehub

    log.info("Fetching Kaggle dataset (tawsifurrahman/covid19-radiography-database)...")
    raw = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")
    nested = os.path.join(raw, "COVID-19_Radiography_Dataset")
    return nested if os.path.isdir(nested) else raw


def _dhash(gray: np.ndarray, size: int = 16) -> int:
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _list_images(source_dir: str) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for ci, cls in enumerate(CLASSES):
        cls_dir = os.path.join(source_dir, cls)
        img_dir = os.path.join(cls_dir, "images") if os.path.isdir(os.path.join(cls_dir, "images")) else cls_dir
        for fn in sorted(os.listdir(img_dir)):
            if fn.lower().endswith(IMG_EXTS):
                entries.append((ci, os.path.join(img_dir, fn)))
    return entries


def build_arrays() -> tuple[np.ndarray, np.ndarray]:
    """
    Load resized grayscale images into a cached .npz file.
    Only contacts Kaggle if no cached .npz exists.
    """
    size = CONFIG["IMG_SIZE"]
    use_clahe = CONFIG["PREPROC"] == "clahe"
    tag = f"cxr_{size}_{'dedup' if CONFIG['DEDUP'] else 'raw'}_{'clahe' if use_clahe else 'plain'}_{K}c"

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = WORK_DIR / f"{tag}.npz"
    if cache.is_file():
        d = np.load(cache)
        log.info("Reusing cached dataset arrays: %s", cache)
        return d["X"], d["y"]

    source_dir = locate_source()
    entries = _list_images(source_dir)
    log.info("Found %d images in %s", len(entries), CLASSES)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None
    X_list, ys, hashes = [], [], []
    for ci, path in entries:
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        hashes.append(_dhash(g) if CONFIG["DEDUP"] else None)
        if clahe is not None:
            g = clahe.apply(g)
        X_list.append(cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA))
        ys.append(ci)

    keep = np.ones(len(ys), dtype=bool)
    if CONFIG["DEDUP"]:
        by_hash: dict[int, set[int]] = {}
        for h, c in zip(hashes, ys):
            by_hash.setdefault(h, set()).add(c)
        seen: set[int] = set()
        dup_same = dup_conflict = 0
        for i, (h, _) in enumerate(zip(hashes, ys)):
            if len(by_hash[h]) > 1:
                keep[i] = False
                dup_conflict += 1
            elif h in seen:
                keep[i] = False
                dup_same += 1
            else:
                seen.add(h)
        log.info("Duplicates removed: %d | label-conflict images removed: %d", dup_same, dup_conflict)

    X = np.stack(X_list)[keep]
    y = np.array(ys, dtype=np.int64)[keep]
    np.savez(cache, X=X, y=y)
    log.info("Saved cache: %s", cache)
    return X, y


def make_split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr_f, va_f, te_f = CONFIG["SPLIT"]
    idx = np.arange(len(y))
    tv, te = train_test_split(idx, test_size=te_f, stratify=y, random_state=CONFIG["SEED"])
    tr, va = train_test_split(tv, test_size=va_f / (tr_f + va_f), stratify=y[tv], random_state=CONFIG["SEED"])
    return tr, va, te


def balance_train(tr: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Balance training split to CONFIG['TRAIN_PER_CLASS'] images per class."""
    per_class = CONFIG["TRAIN_PER_CLASS"]
    if not per_class:
        return tr
    rng = np.random.RandomState(CONFIG["SEED"])
    parts = []
    for k in range(K):
        ids = tr[y[tr] == k]
        parts.append(rng.choice(ids, per_class, replace=len(ids) < per_class))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


# ===========================================================================
# 2. DATASET & GPU AUGMENTATION
# ===========================================================================

class BatchGPUAugment:
    """
    Baseline-protocol augmentation from the paper: rotation +-20 deg, horizontal flip,
    zoom, and gamma contrast applied independently per image on GPU.
    ASSUMPTION: zoom 0.9-1.1 and gamma 0.8-1.2 (magnitudes not stated in paper).
    """

    @torch.no_grad()
    def __call__(self, xb: torch.Tensor) -> torch.Tensor:
        bsz = xb.size(0)
        out = torch.empty_like(xb)
        flips = torch.rand(bsz, device=xb.device) < 0.5
        angles = torch.empty(bsz).uniform_(-20.0, 20.0).tolist()
        scales = torch.empty(bsz).uniform_(0.9, 1.1).tolist()
        for i in range(bsz):
            img = TF.hflip(xb[i]) if flips[i] else xb[i]
            out[i] = TF.affine(img, angle=angles[i], translate=[0, 0], scale=scales[i], shear=[0.0, 0.0])
        out.clamp_(0.0, 1.0)
        g = 0.8 + 0.4 * torch.rand((bsz, 1, 1, 1), device=xb.device)
        return out.pow_(g)


class CXRDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray, gamma: float | None):
        self.X = X
        self.y = y
        self.idx = np.asarray(idx)
        self.gamma = gamma

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        j = self.idx[i]
        t = torch.from_numpy(self.X[j].copy()).float().div_(255.0).unsqueeze(0)
        if self.gamma:
            t.pow_(self.gamma)
        mn, mx = t.min(), t.max()
        t = (t - mn) / (mx - mn + 1e-8)
        return t, int(self.y[j])


def make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    gamma = CONFIG["GAMMA"] if CONFIG["PREPROC"] == "paper" else None
    kw = dict(num_workers=CONFIG["NUM_WORKERS"], pin_memory=(device.type == "cuda"))
    return (
        DataLoader(CXRDataset(X, y, train_idx, gamma), batch_size=batch, shuffle=True, **kw),
        DataLoader(CXRDataset(X, y, val_idx, gamma), batch_size=batch, shuffle=False, **kw),
        DataLoader(CXRDataset(X, y, test_idx, gamma), batch_size=batch, shuffle=False, **kw),
    )


# ===========================================================================
# 3. TRAINING & EVALUATION ENGINE
# ===========================================================================

_scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
_gpu_aug = BatchGPUAugment()
RESULTS: dict[str, dict[str, float]] = {}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    augment: bool = False,
) -> tuple[float, float]:
    is_train = optimizer is not None
    rgb = is_rgb_model(model)
    model.train(is_train)
    loss_sum, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if is_train and augment:
                xb = _gpu_aug(xb)
            if rgb and xb.shape[1] == 1:
                xb = (xb.repeat(1, 3, 1, 1) - IMAGENET_MEAN_DEV) / IMAGENET_STD_DEV

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(xb)
                if isinstance(out, tuple):
                    out = out[0]
                loss = criterion(out, yb)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                _scaler.scale(loss).backward()
                _scaler.step(optimizer)
                _scaler.update()

            loss_sum += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            n += xb.size(0)

    return loss_sum / max(n, 1), correct / max(n, 1)


def fit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    patience: int | None = None,
    plateau: tuple[float, int] | None = None,
    tag: str = "",
    augment: bool = False,
) -> None:
    sched = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=plateau[0], patience=plateau[1])
        if plateau
        else None
    )
    best_loss, best_state, bad = float("inf"), None, 0

    for ep in range(epochs):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, augment=augment)
        vl, va = run_epoch(model, val_loader, criterion, augment=False)
        if sched is not None:
            sched.step(vl)

        lr = optimizer.param_groups[0]["lr"]
        log.info(
            "[%s] epoch %02d/%d | train loss %.4f acc %.2f%% | val loss %.4f acc %.2f%% | lr %.1e | %.0fs",
            tag, ep + 1, epochs, tl, ta * 100, vl, va * 100, lr, time.time() - t0,
        )

        if patience is not None:
            if vl < best_loss - 1e-4:
                best_loss, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    log.info("[%s] early stopping at epoch %d", tag, ep + 1)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    rgb = is_rgb_model(model)
    model.eval()
    probs, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        if rgb and xb.shape[1] == 1:
            xb = (xb.repeat(1, 3, 1, 1) - IMAGENET_MEAN_DEV) / IMAGENET_STD_DEV
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(xb)
            if isinstance(out, tuple):
                out = out[0]
        probs.append(torch.softmax(out.float(), dim=1).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def evaluate_and_print(name: str, probs: np.ndarray, y_true: np.ndarray) -> None:
    pred = probs.argmax(1)
    cm = confusion_matrix(y_true, pred, labels=list(range(K)))
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    sens = tp / np.maximum(tp + fn, 1)
    spec = tn / np.maximum(tn + fp, 1)
    prec = tp / np.maximum(tp + fp, 1)
    f1 = 2 * prec * sens / np.maximum(prec + sens, 1e-12)
    acc = tp.sum() / max(cm.sum(), 1)

    p64 = probs.astype(np.float64)
    p64 /= np.maximum(p64.sum(1, keepdims=True), 1e-12)
    macro_auc = roc_auc_score(y_true, p64, multi_class="ovr", average="macro", labels=list(range(K)))

    print(f"\n=== {name} (test set, n={len(y_true)}) ===")
    print(f"{'class':<18}{'sens':>8}{'spec':>8}{'prec':>8}{'F1':>8}")
    for k in range(K):
        print(f"{CLASSES[k]:<18}{sens[k] * 100:8.2f}{spec[k] * 100:8.2f}{prec[k] * 100:8.2f}{f1[k] * 100:8.2f}")
    print(
        f"accuracy {acc * 100:.2f} | macro sens {sens.mean() * 100:.2f} | macro spec {spec.mean() * 100:.2f} | "
        f"macro prec {prec.mean() * 100:.2f} | macro F1 {f1.mean() * 100:.2f} | macro AUC {macro_auc:.4f}"
    )
    print("Confusion matrix (rows = true, cols = predicted):\n", cm)

    RESULTS[name] = dict(
        acc=acc * 100,
        sens=sens.mean() * 100,
        spec=spec.mean() * 100,
        prec=prec.mean() * 100,
        f1=f1.mean() * 100,
        auc=float(macro_auc),
    )


# ===========================================================================
# 4. TRAINING RUNS (A, B, C)
# ===========================================================================

def train_paper_cnn(optim_name: str, loaders: tuple[DataLoader, DataLoader, DataLoader]) -> nn.Module:
    set_seed(CONFIG["SEED"])
    tr, va, te = loaders
    model = PaperCNN(K, CONFIG["IMG_SIZE"], CONFIG["USE_GAP"]).to(device)
    crit = nn.CrossEntropyLoss()

    if optim_name == "sgdm":
        opt: torch.optim.Optimizer = torch.optim.SGD(
            model.parameters(),
            lr=CONFIG["CNN_LR"],
            momentum=CONFIG["CNN_MOMENTUM"],
            weight_decay=CONFIG["CNN_L2"],
        )
    else:
        opt = torch.optim.Adam(
            model.parameters(),
            lr=CONFIG["CNN_LR"],
            betas=(0.9, 0.999),
            weight_decay=CONFIG["CNN_L2"],
        )

    fit(model, opt, CONFIG["CNN_EPOCHS"], tr, va, crit, tag=f"PaperCNN-{optim_name}", augment=False)
    probs, ys = predict_loader(model, te)
    prefix = "A" if optim_name == "sgdm" else "B"
    evaluate_and_print(f"{prefix}: PaperCNN + {optim_name.upper()}", probs, ys)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / f"paper_cnn_{optim_name}.pth"
    torch.save(model.state_dict(), out_path)
    log.info("Saved checkpoint: %s", out_path)
    return model


def train_baseline(name: str, loaders: tuple[DataLoader, DataLoader, DataLoader]) -> nn.Module:
    set_seed(CONFIG["SEED"])
    tr, va, te = loaders
    model, head = build_pretrained(name, K, CONFIG["PRETRAINED"])
    model.to(device)
    crit = nn.CrossEntropyLoss()
    plateau = (CONFIG["PLATEAU_FACTOR"], CONFIG["PLATEAU_PATIENCE"])

    # Stage 1: backbone frozen, train classification head only
    for p in model.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(head.parameters(), lr=CONFIG["STAGE1_LR"], betas=(0.9, 0.999))
    fit(
        model, opt, CONFIG["STAGE1_EPOCHS"], tr, va, crit,
        patience=CONFIG["ES_PATIENCE"], plateau=plateau, tag=f"{name}-stage1", augment=True,
    )

    # Stage 2: unfreeze full network at lower LR
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["STAGE2_LR"], betas=(0.9, 0.999))
    fit(
        model, opt, CONFIG["STAGE2_EPOCHS"], tr, va, crit,
        patience=CONFIG["ES_PATIENCE"], plateau=plateau, tag=f"{name}-stage2", augment=True,
    )

    probs, ys = predict_loader(model, te)
    evaluate_and_print(f"C: {name} (paper baseline protocol)", probs, ys)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / f"{name}_paper_protocol.pth"
    torch.save(model.state_dict(), out_path)
    log.info("Saved checkpoint: %s", out_path)
    return model


# ===========================================================================
# 5. PAPER FIGURES (Confusion Matrix, ROC, Grad-CAM)
# ===========================================================================

def save_figures(
    probs_sgdm: np.ndarray,
    probs_adam: np.ndarray,
    probs_dense: np.ndarray,
    y_true: np.ndarray,
    m_dense: nn.Module,
    test_loader: DataLoader,
    show: bool = False,
) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (title, preds) in zip(
        axes,
        [
            ("A: PaperCNN + SGDM", probs_sgdm.argmax(1)),
            ("B: PaperCNN + Adam", probs_adam.argmax(1)),
            ("C: DenseNet-121 (Baseline)", probs_dense.argmax(1)),
        ],
    ):
        cm = confusion_matrix(y_true, preds, labels=list(range(K)))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=CLASSES, yticklabels=CLASSES, annot_kws={"size": 11, "weight": "bold"},
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label" if ax is axes[0] else "", fontsize=10)
        ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_confusion_matrices.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    # Figure 2: Multi-Class ROC Curves (DenseNet-121)
    fig = plt.figure(figsize=(8, 6))
    colors = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d"]
    for k in range(K):
        fpr, tpr, _ = roc_curve((y_true == k).astype(int), probs_dense[:, k])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[k % len(colors)], lw=2.2, label=f"{CLASSES[k]} (AUC = {roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title("Multi-Class ROC Curves — DenseNet-121 (Test Set)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_roc_curves.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    # Figure 3: Grad-CAM Heatmaps (one correctly classified test X-ray per class)
    fig, axes = plt.subplots(2, K, figsize=(4 * K, 8))
    test_ds = test_loader.dataset
    for k in range(K):
        mask = (y_true == k) & (probs_dense.argmax(1) == k)
        if not mask.any():
            log.warning("No correct prediction for class %s — skipping Grad-CAM.", CLASSES[k])
            continue
        idx_k = int(np.where(mask)[0][0])
        img_t, _ = test_ds[idx_k]
        cam, probs_1, _ = predict_with_gradcam(m_dense, img_t, target_class=k)
        overlay = build_heatmap_overlay(img_t, cam, alpha=0.45)

        axes[0, k].imshow(img_t.squeeze().numpy(), cmap="gray")
        axes[0, k].set_title(f"Original: {CLASSES[k]}", fontsize=11, fontweight="bold")
        axes[0, k].axis("off")

        axes[1, k].imshow(overlay)
        axes[1, k].set_title(f"Grad-CAM (Conf: {probs_1[k] * 100:.1f}%)", fontsize=10)
        axes[1, k].axis("off")

    plt.suptitle("DenseNet-121 Visual Explanations (Grad-CAM on Test X-Rays)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_gradcam_heatmaps.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    log.info("All figures saved to: %s", FIGURES_DIR)


# ===========================================================================
# 6. CLI ENTRY POINT
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate COVID-19 CXR models")
    parser.add_argument(
        "--run",
        choices=["all", "cnn", "baseline", "figures"],
        default="all",
        help="Which stage(s) to run (default: all)",
    )
    parser.add_argument("--show", action="store_true", help="Display matplotlib figure windows")
    args = parser.parse_args()

    log.info("Device: %s | Classes: %s", device, CLASSES)
    log.info("Models dir: %s | Figures dir: %s", MODELS_DIR, FIGURES_DIR)

    X, y = build_arrays()
    log.info("Dataset loaded: %d images | per-class: %s", len(y), {CLASSES[k]: int((y == k).sum()) for k in range(K)})

    train_idx, val_idx, test_idx = make_split(y)
    train_idx_bal = balance_train(train_idx, y)
    log.info(
        "Split: train %d (balanced %d) | val %d | test %d",
        len(train_idx), len(train_idx_bal), len(val_idx), len(test_idx),
    )

    cnn_loaders = make_loaders(X, y, train_idx_bal, val_idx, test_idx, CONFIG["CNN_BATCH"])
    base_loaders = make_loaders(X, y, train_idx_bal, val_idx, test_idx, CONFIG["BASE_BATCH"])

    m_sgdm = m_adam = None
    baseline_models: dict[str, nn.Module] = {}

    if args.run in ("all", "cnn"):
        m_sgdm = train_paper_cnn("sgdm", cnn_loaders)
        m_adam = train_paper_cnn("adam", cnn_loaders)

    if args.run in ("all", "baseline"):
        for name in CONFIG["BASELINES"]:
            baseline_models[name] = train_baseline(name, base_loaders)

    if RESULTS:
        print(f"\n{'run':<42}{'acc':>7}{'sens':>7}{'spec':>7}{'prec':>7}{'F1':>7}{'AUC':>8}")
        for name, r in RESULTS.items():
            print(f"{name:<42}{r['acc']:7.2f}{r['sens']:7.2f}{r['spec']:7.2f}{r['prec']:7.2f}{r['f1']:7.2f}{r['auc']:8.4f}")
        r = PAPER_REF
        print(f"{'Paper (Table 7, proposed CNN)':<42}{r['acc']:7.2f}{r['sens']:7.2f}{r['spec']:7.2f}{r['prec']:7.2f}{r['f1']:7.2f}{'   -':>8}")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = MODELS_DIR / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CONFIG.items()},
                    "results": RESULTS,
                },
                f,
                indent=2,
            )
        log.info("Saved metrics: %s", results_path)

    if args.run in ("all", "figures"):
        if m_sgdm is None:
            m_sgdm = load_model("paper_cnn", "paper_cnn_sgdm.pth")
        if m_adam is None:
            m_adam = load_model("paper_cnn", "paper_cnn_adam.pth")
        m_dense = baseline_models.get("densenet121") or load_model("densenet121", "densenet121_paper_protocol.pth")

        probs_sgdm, y_true = predict_loader(m_sgdm, cnn_loaders[2])
        probs_adam, _ = predict_loader(m_adam, cnn_loaders[2])
        probs_dense, _ = predict_loader(m_dense, base_loaders[2])

        save_figures(probs_sgdm, probs_adam, probs_dense, y_true, m_dense, base_loaders[2], show=args.show)


if __name__ == "__main__":
    main()
