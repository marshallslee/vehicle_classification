import os, argparse, shutil, random
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from scipy.io import loadmat

# 이미지 사이즈는 224로 고정한다.
IMG_SIZE = 224

def safe_makedirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_or_crop(
    src_img: Path,
    dst_img: Path,
    bbox: Optional[Tuple[int,int,int,int]] = None,
):
    img = cv2.imread(str(src_img))
    if img is None:
        return
    if bbox:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1] - 1, x2), min(img.shape[0] - 1, y2)
        if x2 > x1 and y2 > y1:
            img = img[y1:y2, x1:x2]
    # ✅ 전역 상수 사용
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    cv2.imwrite(str(dst_img), img)

# ---------------- Stanford Cars ----------------
def stanford_lists(stanford_root: Path) -> Tuple[List[Path], List[Path]]:
    image_root = stanford_root / "image"
    tts = stanford_root / "train_test_split"
    images = sorted([p for p in image_root.rglob("*.jpg")])

    if tts.exists():
        train_file = None; test_file = None
        for cand in [("train.txt", "test.txt"), ("train", "test")]:
            if (tts / cand[0]).exists() and (tts / cand[1]).exists():
                train_file, test_file = tts / cand[0], tts / cand[1]
                break

        def read_list(f: Path):
            lst = []
            for line in open(f, "r", encoding="utf-8", errors="ignore"):
                s = line.strip().strip("/")
                if not s:
                    continue
                if s.endswith(".jpg"):
                    lst.append((stanford_root / "image" / s))
                else:
                    lst.extend((stanford_root / "image").rglob(s + ".jpg"))
            return [p for p in lst if p.exists()]

        if train_file and test_file:
            return read_list(train_file), read_list(test_file)

    random.shuffle(images)
    n = len(images); n_train = int(n * 0.8)
    return images[:n_train], images[n_train:]

def stanford_bbox_for_image(
    img_path: Path,
    stanford_root: Path
) -> Optional[Tuple[int,int,int,int]]:
    try:
        rel = img_path.relative_to(stanford_root / "image")
        label_txt = stanford_root / "label" / rel.with_suffix(".txt")
        if not label_txt.exists():
            return None
        lines = [
            l.strip()
            for l in open(label_txt, "r", encoding="utf-8", errors="ignore").read().splitlines()
            if l.strip()
        ]
        if len(lines) >= 3:
            coords = lines[2].split()
            if len(coords) == 4:
                x1, y1, x2, y2 = map(int, coords)
                return x1, y1, x2, y2
    except Exception:
        pass
    return None

def prepare_from_stanford(
    stanford_root: str,
    out: str,
    crop_bbox: bool,
):
    sr = Path(stanford_root); out = Path(out)
    train_imgs, test_imgs = stanford_lists(sr)

    def save_split(imgs: List[Path], split: str):
        dst_dir = out / split / "car"
        safe_makedirs(dst_dir)
        for p in imgs:
            dst = dst_dir / (p.stem + ".jpg")
            bbox = stanford_bbox_for_image(p, sr) if crop_bbox else None
            # ✅ IMG_SIZE는 내부에서 사용
            copy_or_crop(p, dst, bbox)

    # train 전체 저장
    save_split(train_imgs, "train")

    # validation을 train에서 15%로 분리
    val_take = int(len(train_imgs) * 0.15)
    val_imgs = train_imgs[:val_take]
    (out / "val" / "car").mkdir(parents=True, exist_ok=True)
    for p in val_imgs:
        dst = out / "val" / "car" / (p.stem + ".jpg")
        bbox = stanford_bbox_for_image(p, sr) if crop_bbox else None
        copy_or_crop(p, dst, bbox)

    # test 저장
    save_split(test_imgs, "test")

# ---------------- CompCars ----------------
def load_compcars_annos(mat_path: Path):
    try:
        mat = loadmat(str(mat_path))
        annos = mat.get("annotations", None)
        if annos is None:
            return None
        results = {}
        arr = annos[0] if annos.ndim == 2 else annos
        for item in arr:
            def ext_field(x, key):
                try:
                    v = x[key]
                    while isinstance(v, np.ndarray):
                        if v.size == 1:
                            v = v.item()
                        else:
                            break
                    if isinstance(v, bytes):
                        v = v.decode("utf-8")
                    return v
                except Exception:
                    return None

            fname = ext_field(item, "fname") or ext_field(item, "file") or ext_field(item, "filename")
            x1 = ext_field(item, "bbox_x1") or ext_field(item, "x1")
            y1 = ext_field(item, "bbox_y1") or ext_field(item, "y1")
            x2 = ext_field(item, "bbox_x2") or ext_field(item, "x2")
            y2 = ext_field(item, "bbox_y2") or ext_field(item, "y2")
            if fname:
                try:
                    bbox = (int(x1), int(y1), int(x2), int(y2)) if None not in (x1, y1, x2, y2) else None
                except Exception:
                    bbox = None
                results[fname] = bbox
        return results
    except Exception:
        return None

def prepare_from_compcars(
    compcars_root: str,
    out: str,
    crop_bbox: bool,
):
    cr = Path(compcars_root); out = Path(out)
    train_dir = cr / "cars_train" / "cars_train"
    test_dir  = cr / "cars_test"  / "cars_test"
    devkit    = cr / "car_devkit" / "devkit"
    train_mat = devkit / "cars_train_annos.mat"
    test_mat  = devkit / "cars_test_annos.mat"

    train_ann = load_compcars_annos(train_mat) if (crop_bbox and train_mat.exists()) else None
    test_ann  = load_compcars_annos(test_mat)  if (crop_bbox and test_mat.exists())  else None

    def copy_dir(src_dir: Path, split: str, ann):
        dst_dir = out / split / "car"
        safe_makedirs(dst_dir)
        for p in sorted(src_dir.glob("*.jpg")):
            dst = dst_dir / p.name
            bbox = ann.get(p.name) if (ann and p.name in ann) else None
            copy_or_crop(p, dst, bbox)

    if train_dir.exists():
        copy_dir(train_dir, "train", train_ann)
    if test_dir.exists():
        copy_dir(test_dir, "test", test_ann)

    # train에서 validation 15% 샘플링
    train_images = sorted((out / "train" / "car").glob("*.jpg"))
    random.shuffle(train_images)
    val_take = int(len(train_images) * 0.15)
    (out / "val" / "car").mkdir(parents=True, exist_ok=True)
    for p in train_images[:val_take]:
        shutil.copy2(p, (out / "val" / "car" / p.name))

# ---------------- add truck/bus ----------------
def add_single_class(
    src_dir: Optional[str],
    out: str,
    split_triplet=(0.7, 0.15, 0.15),
    cls_name="truck",
):
    """
    src_dir 이하에서 대소문자 무관하게 다양한 확장자(*.jpg, *.jpeg, *.png, *.bmp, *.webp, *.tif, *.tiff)를 재귀 수집.
    IMG_SIZE로 리사이즈해서 out/train|val|test/cls_name 에 저장.
    """
    if not src_dir:
        return
    src = Path(src_dir); out = Path(out)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    # 재귀적으로 다양한 확장자 수집 (대소문자 무시)
    imgs = []
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            imgs.append(p)

    random.shuffle(imgs)
    n = len(imgs)
    if n == 0:
        print(f"[WARN] No images found for {cls_name} in {src_dir}. Supported: {sorted(exts)}")
        return

    n_tr = int(n * split_triplet[0]); n_va = int(n * split_triplet[1])
    parts = [
        ("train", imgs[:n_tr]),
        ("val",   imgs[n_tr:n_tr + n_va]),
        ("test",  imgs[n_tr + n_va:]),
    ]

    for split, lst in parts:
        dst_dir = out / split / cls_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in lst:
            img = cv2.imread(str(p))
            if img is None:
                continue
            # ✅ 전역 IMG_SIZE 사용
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # 이름 충돌 방지: 원본 이름 + 해시 일부
            stem = p.stem
            suffix = ".jpg"  # 최종 jpg로 저장
            safe_name = f"{stem}_{abs(hash(str(p))) % (10**8)}{suffix}"
            cv2.imwrite(str(dst_dir / safe_name), img)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stanford_root", type=str, default=None)
    ap.add_argument("--compcars_root", type=str, default=None)
    ap.add_argument("--truck_dir", type=str, default=None, help="사용자 수집 화물차 이미지 폴더")
    ap.add_argument("--bus_dir", type=str, default=None, help="사용자 수집 버스 이미지 폴더")
    ap.add_argument("--out", type=str, required=True)
    # ✅ img_size 인자 삭제
    ap.add_argument("--crop_bbox", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); safe_makedirs(out)

    print(f"[INFO] Using IMG_SIZE = {IMG_SIZE}")

    if args.stanford_root:
        print("[Stanford] preparing...")
        prepare_from_stanford(args.stanford_root, args.out, args.crop_bbox)

    if args.compcars_root:
        print("[CompCars] preparing...")
        prepare_from_compcars(args.compcars_root, args.out, args.crop_bbox)

    if args.truck_dir:
        print("[Truck] adding...")
        add_single_class(args.truck_dir, args.out, cls_name="truck")
    if args.bus_dir:
        print("[Bus] adding...")
        add_single_class(args.bus_dir, args.out, cls_name="bus")

    print("✅ Done. Output at:", out)

if __name__ == "__main__":
    main()
