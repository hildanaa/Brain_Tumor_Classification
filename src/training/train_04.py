import os, json
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from src.data.dataloaders_02 import create_loaders, LoaderConfig
from src.models.model_factory_03 import build_resnet18, set_trainable_head_only, unfreeze_layer4_and_head

def run_epoch(model, loader, loss_fn, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    y_true, y_pred = [], []

    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(x)
        loss = loss_fn(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    return avg_loss, acc, f1_macro

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ajusta data_root si tus imágenes NO están en data/raw.
    # IMPORTANTE: data_root debe contener Training/ y Testing/ como subcarpetas.
    loader_cfg = LoaderConfig(
      image_size=224,
      batch_size=32,
      num_workers=2,
      data_root="Brain Tumor data"
  )

    train_csv = "data/splits/train.csv"
    val_csv   = "data/splits/val.csv"
    test_csv  = "data/splits/test.csv"

    train_loader, val_loader, test_loader = create_loaders(train_csv, val_csv, test_csv, loader_cfg)

    model, _ = build_resnet18(num_classes=4, pretrained=True)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    # ---- Phase 1: head only ----
    set_trainable_head_only(model)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    best_f1 = -1.0
    best_path = "outputs/checkpoints/best_model.pt"
    log = []

    for epoch in range(1, 6):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)

        row = {"epoch": epoch, "phase": 1, "train_loss": tr_loss, "val_loss": va_loss,
               "train_acc": tr_acc, "val_acc": va_acc, "train_f1_macro": tr_f1, "val_f1_macro": va_f1}
        log.append(row)
        print(row)

        if va_f1 > best_f1:
            best_f1 = va_f1
            os.makedirs("outputs/checkpoints", exist_ok=True)
            torch.save({"model_state": model.state_dict()}, best_path)

    # ---- Phase 2: unfreeze layer4 + head ----
    unfreeze_layer4_and_head(model)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    for epoch in range(6, 11):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)

        row = {"epoch": epoch, "phase": 2, "train_loss": tr_loss, "val_loss": va_loss,
               "train_acc": tr_acc, "val_acc": va_acc, "train_f1_macro": tr_f1, "val_f1_macro": va_f1}
        log.append(row)
        print(row)

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({"model_state": model.state_dict()}, best_path)

    # ---- Test eval with best ----
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    te_loss, te_acc, te_f1 = run_epoch(model, test_loader, loss_fn, optimizer=None, device=device, train=False)

    metrics = {
        "best_val_f1_macro": best_f1,
        "test_loss": te_loss,
        "test_acc": te_acc,
        "test_f1_macro": te_f1,
        "device": device
    }
    save_json("outputs/metrics/metrics.json", metrics)
    save_json("outputs/metrics/train_log.json", log)

    print("Saved:", best_path, "and outputs/metrics/metrics.json")

if __name__ == "__main__":
    main()