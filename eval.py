import os, json, argparse, torch, numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import build_model

IMG_SIZE = 224
BATCH_SIZE = 64


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--arch", type=str, required=True)
    p.add_argument("--num_workers", type=int, default=4)

    return p.parse_args()


def plot_confusion(cm, class_names, out_png):
    import numpy as np
    fig = plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.ylabel('True');
    plt.xlabel('Pred')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)

    arch = args.arch
    class_to_idx = ckpt["class_to_idx"]
    class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]

    model = build_model(
        arch,
        num_classes=len(class_to_idx)
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    test_ds = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"),
        transform=tf
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers
    )

    preds = []
    gts = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            gts.extend(yb.cpu().tolist())

    acc = accuracy_score(gts, preds)
    cm = confusion_matrix(gts, preds)
    rep = classification_report(gts, preds, target_names=class_names, digits=4)

    print("Test Accuracy:", acc)
    print(rep)

    out_dir = os.path.dirname(args.checkpoint)

    plot_confusion(cm, class_names, os.path.join(out_dir, "confusion_matrix_eval.png"))

    json.dump({"test_accuracy": acc, "report": rep}, open(os.path.join(out_dir, "metrics_eval.json"), "w"))