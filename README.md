# Medical Diagnosis — COVID-19 Chest X-Ray Classifier

Clean modular implementation of the Sci Rep 2026 paper (PMC13470498) for classifying Chest X-Rays into **COVID**, **Lung Opacity**, **Normal**, and **Viral Pneumonia** with **Grad-CAM** visual explanations.

---

## Folder Structure

```text
medical diagnosis/
├── model.py        ← The Model: config, PaperCNN & DenseNet-121 architectures, preprocessing, Grad-CAM, checkpoint loader
├── app.py          ← Use the Model: Gradio medical diagnosis web app + CLI single-image tool
├── train.py        ← Training & Evaluation: dataset loading, Runs A/B/C training, metrics, paper figures
├── models/         ← Trained weights (.pth) + results.json (ready to use — no retraining needed!)
│   ├── densenet121_paper_protocol.pth
│   ├── paper_cnn_adam.pth
│   ├── paper_cnn_sgdm.pth
│   └── results.json
├── figures/        ← Saved evaluation figures (Confusion Matrix, ROC Curves, Grad-CAM)
└── README.md
```

---

## Quick Start (Use the Trained Model Immediately)

No retraining is needed — your trained checkpoints are already inside `models/`.

### Option 1 — Interactive Web App (Gradio)

```powershell
cd "C:\Users\Sigma\OneDrive\Desktop\medical diagnosis"
python app.py
```

This opens the web UI in your browser at `http://127.0.0.1:7860`:
- Upload any chest X-ray image (PNG / JPG / BMP / TIFF — RGB, RGBA, or grayscale)
- Switch between **DenseNet-121 (93.7%)**, **PaperCNN + Adam (85.7%)**, and **PaperCNN + SGDM (79.4%)**
- Adjust the **Grad-CAM Opacity** slider live
- View probability bars, Grad-CAM heatmap, and bilingual (English + Arabic) summary

Extra options:
```powershell
python app.py --share              # Generate a public Gradio share link
python app.py --port 8000          # Use a different port
```

### Option 2 — Command-Line Single Image Diagnosis

```powershell
python app.py --image "path\to\xray.png" --save-cam "gradcam_out.png"
```

---

## Re-Training or Regenerating Figures (Optional)

Only run `train.py` if you want to retrain the models or regenerate plots:

```powershell
python train.py                   # Train Runs A, B, C + regenerate figures
python train.py --run cnn         # Train only PaperCNN (SGDM & Adam)
python train.py --run baseline    # Train only DenseNet-121 fine-tuning
python train.py --run figures     # Regenerate Confusion Matrix / ROC / Grad-CAM from saved weights
```

---

## Evaluation Results (Test Set)

| Model | Accuracy | Macro F1 | Macro AUC |
|-------|----------|----------|-----------|
| A: PaperCNN + SGDM | 79.41% | 80.25% | 0.9500 |
| B: PaperCNN + Adam | 85.67% | 86.59% | 0.9680 |
| **C: DenseNet-121** | **93.69%** | **94.78%** | **0.9925** |
| Paper (Table 7, reported) | 97.51% | 86.91% | — |

---

> ⚠️ **Disclaimer:** This project is for research and educational use only and is not a certified medical diagnostic tool.
