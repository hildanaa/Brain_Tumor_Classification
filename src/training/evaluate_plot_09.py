import os
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, f1_score, accuracy_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from src.data.dataloaders_02 import create_loaders, LoaderConfig
from src.models.model_factory_03 import build_resnet18

LABELS = ["glioma", "meningioma", "pituitary", "notumor"]  # tu mapping real

def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model

@torch.no_grad()
def predict_loader(model, loader, device):
    y_true = []
    y_pred = []
    y_prob = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        y_true.extend(y.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.append(probs)

    y_prob = np.vstack(y_prob)
    return np.array(y_true), np.array(y_pred), y_prob

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def plot_training_curves(train_log_path, out_dir):
    ensure_dir(out_dir)
    with open(train_log_path, "r", encoding="utf-8") as f:
        log = json.load(f)

    epochs = [r["epoch"] for r in log]
    tr_loss = [r["train_loss"] for r in log]
    va_loss = [r["val_loss"] for r in log]
    tr_acc  = [r["train_acc"] for r in log]
    va_acc  = [r["val_acc"] for r in log]
    tr_f1   = [r["train_f1_macro"] for r in log]
    va_f1   = [r["val_f1_macro"] for r in log]

    # Loss
    plt.figure()
    plt.plot(epochs, tr_loss, label="train_loss")
    plt.plot(epochs, va_loss, label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.title("Training vs Validation Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_loss.png"))
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, tr_acc, label="train_acc")
    plt.plot(epochs, va_acc, label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
    plt.title("Training vs Validation Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_accuracy.png"))
    plt.close()

    # Macro F1
    plt.figure()
    plt.plot(epochs, tr_f1, label="train_f1_macro")
    plt.plot(epochs, va_f1, label="val_f1_macro")
    plt.xlabel("epoch"); plt.ylabel("macro F1"); plt.legend()
    plt.title("Training vs Validation Macro-F1")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_f1_macro.png"))
    plt.close()

def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    plt.figure()
    disp.plot(values_format="d")
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_f1_by_class(y_true, y_pred, out_path):
    # F1 por clase
    f1s = f1_score(y_true, y_pred, average=None)
    plt.figure()
    plt.bar(LABELS, f1s)
    plt.ylim(0, 1.0)
    plt.ylabel("F1 score")
    plt.title("F1 by Class (Test)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_roc_ovr(y_true, y_prob, out_path):
    # One-vs-rest ROC
    y_true_bin = label_binarize(y_true, classes=list(range(len(LABELS))))
    plt.figure()

    aucs = []
    for i, label in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")

    plt.plot([0,1], [0,1], linestyle="--", label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC One-vs-Rest (macro AUC={np.mean(aucs):.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ajusta a tu estructura real (por tus screenshots)
    loader_cfg = LoaderConfig(
        image_size=224,
        batch_size=32,
        num_workers=0,          # recomendado en Windows
        data_root="Brain Tumor data"
    )

    train_csv = "data/splits/train.csv"
    val_csv   = "data/splits/val.csv"
    test_csv  = "data/splits/test.csv"

    _, _, test_loader = create_loaders(train_csv, val_csv, test_csv, loader_cfg)

    model, _ = build_resnet18(num_classes=4, pretrained=False)
    model = load_checkpoint(model, "outputs/checkpoints/best_model.pt", device)

    y_true, y_pred, y_prob = predict_loader(model, test_loader, device)

    # Métricas numéricas
    metrics = {
        "test_acc": float(accuracy_score(y_true, y_pred)),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred, target_names=LABELS, output_dict=True),
        "device": device
    }

    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/test_report.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    os.makedirs("outputs/figures", exist_ok=True)
    plot_confusion(y_true, y_pred, "outputs/figures/confusion_matrix_test.png")
    plot_f1_by_class(y_true, y_pred, "outputs/figures/f1_by_class_test.png")

    # Curvas de training (si existe train_log.json)
    if os.path.exists("outputs/metrics/train_log.json"):
        plot_training_curves("outputs/metrics/train_log.json", "outputs/figures")

    # ROC (opcional, pero te lo dejo prendido)
    plot_roc_ovr(y_true, y_prob, "outputs/figures/roc_ovr_test.png")

    print("Saved plots to outputs/figures and report to outputs/metrics/test_report.json")

if __name__ == "__main__":
    main()