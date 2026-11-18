"""
model.py — Shared configuration, neural network architectures, preprocessing,
Grad-CAM explainability, and checkpoint loading for COVID-19 Chest X-Ray classification.

Replicates the Sci Rep 2026 paper (PMC13470498):
  • PaperCNN     — 19-layer CNN from scratch (Table 3)
  • DenseNet-121 — Pretrained transfer-learning baseline (best accuracy: 93.69%)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ===========================================================================
# 1. CONFIGURATION & PATHS
# ===========================================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR  = PROJECT_DIR / "models"
FIGURES_DIR = PROJECT_DIR / "figures"

# Fallback directory for dataset cache (.npz) or legacy Colab outputs
_DEFAULT_WORK = r"C:\content" if os.name == "nt" else "/content"
WORK_DIR = Path(os.environ.get("CXR_WORK_DIR", _DEFAULT_WORK))
LEGACY_OUT_DIR = WORK_DIR / "cxr_paper_outputs"

CONFIG = dict(
    # ---- Data ----------------------------------------------------------------
    USE_LUNG_OPACITY = True,    # True = paper's 4-class setup (21,165 images); False = 3-class
    IMG_SIZE         = 227,     # Paper Table 3 input layer: 227x227x1
    DEDUP            = True,    # NOT in paper: removes near-duplicates to limit leakage
    PREPROC          = "paper", # "paper" = gamma + min-max | "clahe" = CLAHE + min-max
    GAMMA            = 0.8,     # ASSUMPTION: paper uses gamma correction but gives no value
    TRAIN_PER_CLASS  = 3500,    # Paper: balanced training set of 3,500 images per class
    SPLIT            = (0.80, 0.10, 0.10),  # Paper: 80/10/10 stratified split
    SEED             = 42,
    NUM_WORKERS      = 0,       # 0 avoids multiprocessing crashes on Windows

    # ---- Paper's proposed 19-layer CNN ---------------------------------------
    USE_GAP          = False,   # Table 3 + 19-layer count imply NO global avg-pool
    CNN_LR           = 1e-4,    # Paper: initial learning rate 0.0001
    CNN_MOMENTUM     = 0.9,     # ASSUMPTION: not stated (MATLAB default)
    CNN_BATCH        = 128,     # ASSUMPTION: not stated (MATLAB default MiniBatchSize)
    CNN_L2           = 1e-4,    # ASSUMPTION: not stated (MATLAB default L2Regularization)
    CNN_EPOCHS       = 6,       # Paper: "almost 6 epochs"

    # ---- Paper's baseline fine-tuning protocol (pretrained networks) ---------
    PRETRAINED       = True,
    BASELINES        = ["densenet121"],  # densenet121, resnet50, vgg19, alexnet, googlenet, squeezenet1_0
    BASE_BATCH       = 64,
    STAGE1_EPOCHS    = 15, STAGE1_LR = 1e-4,  # Stage 1: backbone frozen, head only
    STAGE2_EPOCHS    = 25, STAGE2_LR = 1e-5,  # Stage 2: full fine-tuning
    ES_PATIENCE      = 5,
    PLATEAU_FACTOR   = 0.5,
    PLATEAU_PATIENCE = 3,
)

CLASSES = (
    ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
    if CONFIG["USE_LUNG_OPACITY"]
    else ["COVID", "Normal", "Viral Pneumonia"]
)
K = len(CLASSES)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ImageNet normalisation tensors (for RGB pretrained backbones)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Paper Table 7 reported metrics (for reference comparison)
PAPER_REF = dict(acc=97.51, sens=88.43, spec=89.60, prec=87.59, f1=86.91)

# Registry of available trained checkpoints
AVAILABLE_MODELS = {
    "DenseNet-121 (Best — 93.7% Acc)": ("densenet121", "densenet121_paper_protocol.pth"),
    "PaperCNN + Adam (85.7% Acc)":     ("paper_cnn",   "paper_cnn_adam.pth"),
    "PaperCNN + SGDM (79.4% Acc)":     ("paper_cnn",   "paper_cnn_sgdm.pth"),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN_DEV = IMAGENET_MEAN.to(device)
IMAGENET_STD_DEV  = IMAGENET_STD.to(device)


# ===========================================================================
# 2. MODEL ARCHITECTURES
# ===========================================================================

class PaperCNN(nn.Module):
    """
    Paper Table 3: 19-layer CNN for grayscale chest X-rays.
    Architecture:
      input -> 3 x [Conv3x3 -> BN -> ReLU -> MaxPool2x2] (8 / 16 / 32 filters)
            -> Conv3x3(64) -> BN -> ReLU -> Flatten -> FC -> softmax
    MATLAB-style layer count: 1 input + 15 + 1 FC + 1 softmax + 1 classification = 19.
    Softmax is folded into nn.CrossEntropyLoss during training.
    """

    def __init__(self, num_classes: int = K, img_size: int = CONFIG["IMG_SIZE"], use_gap: bool = CONFIG["USE_GAP"]):
        super().__init__()

        def block(cin: int, cout: int, pool: bool) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            return layers

        self.features = nn.Sequential(
            *block(1, 8, True),
            *block(8, 16, True),
            *block(16, 32, True),
            *block(32, 64, False),
        )
        self.gap = nn.AdaptiveAvgPool2d(1) if use_gap else nn.Identity()

        spatial = img_size // 8  # 3 max-pooling layers (2^3 = 8)
        fc_in = 64 if use_gap else 64 * spatial * spatial
        self.fc = nn.Linear(fc_in, num_classes)

        # MATLAB default init: Glorot (Xavier) uniform weights, zero bias
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.gap(self.features(x)), 1))


def build_pretrained(name: str, num_classes: int = K, pretrained: bool = True) -> tuple[nn.Module, nn.Module]:
    """Build a torchvision backbone with a new classification head for `num_classes`."""
    weights = "DEFAULT" if pretrained else None
    if name == "densenet121":
        m = models.densenet121(weights=weights)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        head = m.classifier
    elif name == "resnet50":
        m = models.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head = m.fc
    elif name in ("vgg19", "alexnet"):
        m = getattr(models, name)(weights=weights)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
        head = m.classifier[6]
    elif name == "googlenet":
        m = models.googlenet(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head = m.fc
    elif name == "squeezenet1_0":
        m = models.squeezenet1_0(weights=weights)
        m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        head = m.classifier[1]
    else:
        raise ValueError(f"Unknown baseline architecture: {name!r}")
    return m, head


def is_rgb_model(model: nn.Module) -> bool:
    """Return True if the model expects 3-channel RGB inputs."""
    first_param = next(model.parameters())
    return first_param.ndim == 4 and first_param.shape[1] == 3


# ===========================================================================
# 3. CHECKPOINT RESOLUTION & LOADING
# ===========================================================================

def resolve_checkpoint(filename: str) -> Path:
    """
    Find a checkpoint file by checking:
      1. Exact path if `filename` already exists on disk
      2. ./models/<filename> inside this project folder
      3. Legacy output directory (C:/content/cxr_paper_outputs/<filename>)
    """
    direct = Path(filename)
    if direct.is_file():
        return direct

    local_path = MODELS_DIR / filename
    if local_path.is_file():
        return local_path

    legacy_path = LEGACY_OUT_DIR / filename
    if legacy_path.is_file():
        return legacy_path

    raise FileNotFoundError(
        f"Model checkpoint '{filename}' not found.\n"
        f"Searched:\n  - {local_path}\n  - {legacy_path}\n"
        f"Run `python train.py` to generate the weights or pass `--model <path>`."
    )


def load_model(arch: str, checkpoint: str | Path, target_device: torch.device = device) -> nn.Module:
    """Instantiate `arch` ('densenet121', 'paper_cnn', etc.) and load weights."""
    ckpt_path = resolve_checkpoint(str(checkpoint))
    if arch == "paper_cnn":
        model: nn.Module = PaperCNN(K, CONFIG["IMG_SIZE"], CONFIG["USE_GAP"])
    else:
        model, _ = build_pretrained(arch, K, pretrained=False)

    state_dict = torch.load(ckpt_path, map_location=target_device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()
    return model


# ===========================================================================
# 4. PREPROCESSING & GRAD-CAM INFERENCE
# ===========================================================================

def preprocess_xray(input_img: np.ndarray, img_size: int = CONFIG["IMG_SIZE"],
                    gamma: Optional[float] = CONFIG["GAMMA"]) -> torch.Tensor:
    """
    Convert any image array (RGB, RGBA, grayscale 2D/3D, uint8/uint16/float)
    into a normalized [1, H, W] float32 tensor matching the training pipeline.
    """
    if input_img is None or input_img.size == 0:
        raise ValueError("Empty input image.")

    img = np.asarray(input_img)

    # Convert high bit-depth or float images cleanly to float32 in [0, 255] for cv2
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
    elif np.issubdtype(img.dtype, np.floating):
        max_val = float(img.max()) if img.size else 1.0
        scale = 255.0 if max_val <= 1.0 else 1.0
        img = np.clip(img * scale, 0, 255).astype(np.uint8)

    # Normalize channels -> 2D grayscale
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 3 and img.shape[2] == 1:
        gray = img[:, :, 0]
    elif img.ndim == 2:
        gray = img
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)

    t = torch.from_numpy(resized.copy()).float().div_(255.0).unsqueeze(0)
    if gamma:
        t = t.pow(gamma)
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)
    return t  # shape: [1, H, W]


def predict_with_gradcam(
    model: nn.Module,
    img_1ch: torch.Tensor,
    target_class: Optional[int] = None,
    target_device: torch.device = device,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Single-pass inference + Grad-CAM heatmap for DenseNet-121 or PaperCNN.

    Returns:
      cam          : [H, W] float32 array in [0, 1]
      probs        : [K] float32 probability array across CLASSES
      target_class : class index used for Grad-CAM (predicted class if None)
    """
    model.eval()
    mean = IMAGENET_MEAN.to(target_device)
    std  = IMAGENET_STD.to(target_device)

    xb = img_1ch.unsqueeze(0).to(target_device)

    with torch.enable_grad():
        if isinstance(model, PaperCNN):
            xb = xb.requires_grad_(True)
            activations = model.features(xb)
            activations.retain_grad()
            out = model.fc(torch.flatten(model.gap(activations), 1))
        elif hasattr(model, "features") and hasattr(model, "classifier"):
            # DenseNet-121: manual forward with non-inplace ReLU avoids autograd
            # RuntimeError ("BackwardHookFunctionBackward is a view and is being modified inplace")
            xb_rgb = ((xb.repeat(1, 3, 1, 1) - mean) / std).requires_grad_(True)
            features = model.features(xb_rgb)
            activations = torch.relu(features)
            activations.retain_grad()
            pooled = F.adaptive_avg_pool2d(activations, (1, 1))
            out = model.classifier(torch.flatten(pooled, 1))
        else:
            raise TypeError(f"Grad-CAM is not implemented for {type(model).__name__}")

        probs_t = torch.softmax(out.detach().float(), dim=1)[0]
        if target_class is None:
            target_class = int(probs_t.argmax().item())

        model.zero_grad(set_to_none=True)
        out[0, target_class].backward()

        weights = activations.grad.mean(dim=(2, 3), keepdim=True)
        cam_t = torch.relu((weights * activations.detach()).sum(dim=1)).squeeze()
        cam = cam_t.cpu().numpy().astype(np.float32)

    cam = cv2.resize(cam, (CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]), interpolation=cv2.INTER_CUBIC)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, probs_t.cpu().numpy(), target_class


def build_heatmap_overlay(img_1ch: torch.Tensor, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend grayscale chest X-ray with JET Grad-CAM heatmap -> uint8 RGB [H, W, 3]."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    orig = img_1ch.squeeze().cpu().numpy()
    orig_rgb = np.stack([orig] * 3, axis=-1)
    heatmap_bgr = cv2.applyColorMap(np.uint8(np.clip(cam, 0.0, 1.0) * 255), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blended = np.clip((1.0 - alpha) * orig_rgb + alpha * heatmap_rgb, 0.0, 1.0)
    return np.uint8(blended * 255)


if __name__ == "__main__":
    _m = PaperCNN()
    n_params = sum(p.numel() for p in _m.parameters())
    print(f"Device: {device} | Classes ({K}): {CLASSES}")
    print(f"PaperCNN parameters: {n_params:,}")
    print(f"Models dir: {MODELS_DIR}")
