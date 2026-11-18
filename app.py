"""
app.py — Medical Diagnosis App for COVID-19 Chest X-Ray Classification.

Uses the trained models from `model.py` (no retraining required) to:
  1. Classify a chest X-ray into COVID / Lung_Opacity / Normal / Viral Pneumonia
  2. Generate a Grad-CAM heatmap showing the lung regions driving the decision
  3. Present a clear confidence breakdown and diagnostic summary

Usage
-----
  python app.py                                # Launch interactive Gradio web app
  python app.py --share                        # Create a shareable Gradio link
  python app.py --port 7860                    # Custom local port
  python app.py --image path/to/xray.png       # Run single-image diagnosis in CLI
  python app.py --image xray.png --save-cam cam.png
"""

from __future__ import annotations

import argparse
import logging
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from model import (
    AVAILABLE_MODELS,
    CLASSES,
    K,
    build_heatmap_overlay,
    device,
    load_model,
    predict_with_gradcam,
    preprocess_xray,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ===========================================================================
# Clinical Class Metadata
# ===========================================================================

CLASS_INFO = {
    "COVID": (
        "COVID-19 Pneumonia",
        "Patterns consistent with COVID-19 viral lung involvement.",
    ),
    "Lung_Opacity": (
        "Non-COVID Lung Opacity",
        "Abnormal lung opacity / consolidation not specific to COVID-19.",
    ),
    "Normal": (
        "Normal Chest X-Ray",
        "No obvious radiographic signs of COVID-19 or pneumonia detected.",
    ),
    "Viral Pneumonia": (
        "Viral Pneumonia",
        "Patterns consistent with non-COVID viral pneumonia.",
    ),
}

DEFAULT_MODEL_KEY = list(AVAILABLE_MODELS.keys())[0]


# ===========================================================================
# Model Cache & Core Inference
# ===========================================================================

class DiagnosisEngine:
    """Lazy-loading, cached inference engine supporting multiple trained models."""

    def __init__(self, custom_checkpoint: str | None = None, custom_arch: str = "densenet121"):
        self._cache: dict[str, torch.nn.Module] = {}
        self.custom_checkpoint = custom_checkpoint
        self.custom_arch = custom_arch

    def get_model(self, model_key: str) -> torch.nn.Module:
        if self.custom_checkpoint:
            cache_key = f"custom::{self.custom_checkpoint}"
            if cache_key not in self._cache:
                log.info("Loading custom checkpoint: %s", self.custom_checkpoint)
                self._cache[cache_key] = load_model(self.custom_arch, self.custom_checkpoint)
            return self._cache[cache_key]

        if model_key not in self._cache:
            arch, filename = AVAILABLE_MODELS[model_key]
            log.info("Loading model '%s' (%s)...", model_key, filename)
            self._cache[model_key] = load_model(arch, filename)
        return self._cache[model_key]

    def diagnose(
        self,
        input_img: np.ndarray,
        model_key: str = DEFAULT_MODEL_KEY,
        heatmap_alpha: float = 0.45,
    ) -> tuple[np.ndarray, dict[str, float], str]:
        """
        Run preprocessing + single-pass inference + Grad-CAM.
        Returns (overlay_rgb_uint8, confidences_dict, markdown_report).
        """
        model = self.get_model(model_key)
        img_tensor = preprocess_xray(input_img)
        cam, probs, pred_idx = predict_with_gradcam(model, img_tensor)
        overlay = build_heatmap_overlay(img_tensor, cam, alpha=heatmap_alpha)

        confidences = {CLASSES[i]: float(probs[i]) for i in range(K)}
        pred_class = CLASSES[pred_idx]
        pred_conf = float(probs[pred_idx]) * 100.0

        en_name, desc = CLASS_INFO.get(pred_class, (pred_class, ""))
        if pred_conf >= 85.0:
            certainty = "High"
        elif pred_conf >= 60.0:
            certainty = "Moderate"
        else:
            certainty = "Low — review carefully"

        report = (
            f"### Predicted Diagnosis: **{en_name}**\n"
            f"- **Confidence:** `{pred_conf:.2f}%` ({certainty})\n"
            f"- **Model:** `{model_key}` on `{device}`\n"
            f"- **Summary:** {desc}\n\n"
            f"> ⚠️ *For research and educational demonstration only — not a substitute for clinical medical advice.*"
        )
        return overlay, confidences, report


# ===========================================================================
# CLI Mode (Single Image Diagnosis)
# ===========================================================================

def run_cli_diagnosis(
    engine: DiagnosisEngine,
    image_path: str,
    model_key: str,
    alpha: float,
    save_cam: str | None,
) -> None:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")

    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError(f"Could not decode image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 and bgr.shape[2] == 3 else bgr

    overlay, confidences, _ = engine.diagnose(rgb, model_key=model_key, heatmap_alpha=alpha)

    print(f"\n=== Medical Diagnosis Result ({model_key}) ===")
    print(f"Image: {path}")
    for cls, score in sorted(confidences.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(score * 30)
        print(f"  {cls:<18} {score * 100:6.2f}%  {bar}")

    if save_cam:
        out_path = Path(save_cam)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved Grad-CAM overlay to: {out_path}")


# ===========================================================================
# Interactive Gradio Web App
# ===========================================================================

def launch_web_app(engine: DiagnosisEngine, port: int | None = None, share: bool = False) -> None:
    import gradio as gr

    # Preload default model at startup so the first prediction is instant
    engine.get_model(DEFAULT_MODEL_KEY)

    model_choices = (
        [f"Custom ({Path(engine.custom_checkpoint).name})"]
        if engine.custom_checkpoint
        else list(AVAILABLE_MODELS.keys())
    )

    def _predict_ui(input_img: np.ndarray | None, model_choice: str, alpha: float):
        if input_img is None:
            return None, {}, "*Upload a chest X-ray image and click **Diagnose** to begin.*"
        try:
            return engine.diagnose(input_img, model_key=model_choice, heatmap_alpha=alpha)
        except Exception as exc:
            traceback.print_exc()
            raise gr.Error(f"Diagnosis failed: {exc}") from exc

    with gr.Blocks(title="COVID-19 Chest X-Ray Medical Diagnosis") as demo:
        gr.Markdown(
            """
            # 🩻 COVID-19 Chest X-Ray Medical Diagnosis
            Upload a chest X-ray image to classify it into **COVID**, **Lung Opacity**, **Normal**, or **Viral Pneumonia**,
            with a **Grad-CAM** heatmap highlighting the regions that influenced the prediction (red = highest attention).
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(label="Upload Chest X-Ray", type="numpy")
                model_dd = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0],
                    label="Model",
                    interactive=len(model_choices) > 1,
                )
                alpha_slider = gr.Slider(
                    minimum=0.0,
                    maximum=0.85,
                    value=0.45,
                    step=0.05,
                    label="Grad-CAM Heatmap Opacity",
                )
                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    submit_btn = gr.Button("Diagnose", variant="primary")

            with gr.Column(scale=1):
                cam_out = gr.Image(label="Grad-CAM Attention Map", type="numpy")
                label_out = gr.Label(num_top_classes=K, label="Diagnosis Probabilities")
                report_out = gr.Markdown(value="*Upload a chest X-ray image and click **Diagnose** to begin.*")

        submit_btn.click(
            fn=_predict_ui,
            inputs=[img_in, model_dd, alpha_slider],
            outputs=[cam_out, label_out, report_out],
        )
        # Re-render live when user adjusts opacity or switches model with an image already uploaded
        alpha_slider.release(
            fn=_predict_ui,
            inputs=[img_in, model_dd, alpha_slider],
            outputs=[cam_out, label_out, report_out],
        )
        model_dd.change(
            fn=_predict_ui,
            inputs=[img_in, model_dd, alpha_slider],
            outputs=[cam_out, label_out, report_out],
        )
        clear_btn.click(
            fn=lambda: (None, None, {}, "*Upload a chest X-ray image and click **Diagnose** to begin.*"),
            outputs=[img_in, cam_out, label_out, report_out],
        )

    port_label = str(port) if port is not None else "auto (7860+)"
    log.info("Launching Medical Diagnosis app (port=%s, device=%s)", port_label, device)
    demo.launch(
        server_port=port,
        share=share,
        show_error=True,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )


# ===========================================================================
# Entry Point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="COVID-19 Chest X-Ray Medical Diagnosis App")
    parser.add_argument("--model", type=str, default=None, help="Optional path to a custom .pth checkpoint")
    parser.add_argument(
        "--arch",
        type=str,
        default="densenet121",
        choices=["densenet121", "paper_cnn"],
        help="Architecture when using --model (default: densenet121)",
    )
    parser.add_argument("--image", type=str, default=None, help="Classify a single X-ray image in CLI mode")
    parser.add_argument("--save-cam", type=str, default=None, help="Save Grad-CAM overlay image (CLI mode)")
    parser.add_argument("--alpha", type=float, default=0.45, help="Grad-CAM heatmap opacity (default: 0.45)")
    parser.add_argument("--port", type=int, default=None, help="Local Gradio port (default: auto-pick free port starting at 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    engine = DiagnosisEngine(custom_checkpoint=args.model, custom_arch=args.arch)

    if args.image:
        model_key = f"Custom ({Path(args.model).name})" if args.model else DEFAULT_MODEL_KEY
        run_cli_diagnosis(engine, args.image, model_key=model_key, alpha=args.alpha, save_cam=args.save_cam)
    else:
        launch_web_app(engine, port=args.port, share=args.share)


if __name__ == "__main__":
    main()
