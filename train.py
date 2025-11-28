import os, time, json, random
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from models import build_model
from data_utils import get_loaders

IMG_SIZE = 224
BATCH_SIZE = 64


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--arch", type=str, default="baseline",
                   choices=["baseline", "resnet50", "efficientnet_b0"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--unfreeze_at", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--class_names", nargs="+", default=["car", "truck", "bus"])
    p.add_argument("--auto_split", action="store_true")
    p.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--weighted_loss", action="store_true")
    p.add_argument("--use_weighted_sampler", action="store_true")
    p.add_argument("--mixup", action="store_true")
    p.add_argument("--mixup_alpha", type=float, default=0.2)

    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.counter = 0
        self.should_stop = False
    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss; self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b), lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def plot_confusion(cm, class_names, out_png):
    fig = plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick = np.arange(len(class_names))
    plt.xticks(tick, class_names, rotation=45)
    plt.yticks(tick, class_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    args = get_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    # ======================================
    # IMG_SIZE, BATCH_SIZE은 고정 상수 사용
    # ======================================
    train_loader, val_loader, test_loader, class_to_idx = get_loaders(
        args.data_dir,
        IMG_SIZE,
        BATCH_SIZE,                 # ← 상수 사용
        args.num_workers,
        args.auto_split,
        args.class_names,
        args.use_weighted_sampler
    )

    num_classes = len(class_to_idx)
    model = build_model(
        args.arch,
        num_classes=num_classes,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    # Loss (Weighted optional)
    criterion = nn.CrossEntropyLoss()
    if args.weighted_loss:
        counts = np.zeros(num_classes, dtype=np.float32)
        for _, y in train_loader.dataset.samples:
            counts[y] += 1
        weights = counts.sum() / (counts + 1e-6)
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs
    )

    early = EarlyStopping(patience=5, min_delta=1e-4)

    run_name = f"{args.arch}_{time.strftime('%Y%m%d-%H%M')}"
    save_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    best_val = float('inf')
    best_path = os.path.join(save_dir, "best.ckpt")

    # ---------------- TRAIN ----------------
    for epoch in range(1, args.epochs + 1):

        if args.unfreeze_at and epoch == args.unfreeze_at:
            for p in model.parameters():
                p.requires_grad = True

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        tr_loss = 0.

        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)

            if args.mixup:
                xb, (ya, yb2), lam = mixup_data(xb, yb, alpha=args.mixup_alpha)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)

            loss = (
                mixup_criterion(criterion, logits, ya, yb2, lam)
                if args.mixup else
                criterion(logits, yb)
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            tr_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        tr_loss /= len(train_loader)

        # ---------------- VALID ----------------
        model.eval()
        val_loss = 0.; preds = []; gts = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
                preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                gts.extend(yb.cpu().tolist())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(gts, preds)

        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        early.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": args.arch,
                    "class_to_idx": class_to_idx
                },
                best_path
            )
        if early.should_stop:
            print("Early stopping triggered.")
            break

    # ---------------- TEST ----------------
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preds = []; gts = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            gts.extend(yb.cpu().tolist())

    acc = accuracy_score(gts, preds)
    cm = confusion_matrix(gts, preds)
    rep = classification_report(gts, preds, target_names=args.class_names, digits=4)

    print("Test Accuracy:", acc)
    print(rep)

    json.dump(
        {"val_best_loss": best_val, "test_accuracy": acc, "report": rep},
        open(os.path.join(save_dir, "metrics.json"), "w")
    )

    plot_confusion(cm, args.class_names, os.path.join(save_dir, "confusion_matrix.png"))


if __name__ == "__main__":
    main()
