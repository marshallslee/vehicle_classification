from __future__ import annotations
import os, random, shutil
from typing import List
from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)], p=0.8
        ),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf

def auto_split_if_needed(root: str, class_names: List[str], split=(0.7,0.15,0.15)):
    train_dir = os.path.join(root, "train")
    val_dir   = os.path.join(root, "val")
    test_dir  = os.path.join(root, "test")
    if all(os.path.isdir(p) for p in [train_dir,val_dir,test_dir]):
        return train_dir, val_dir, test_dir

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for cname in class_names:
        src = os.path.join(root, cname)
        imgs = [os.path.join(src,f) for f in os.listdir(src)
                if os.path.isfile(os.path.join(src,f))]
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n*split[0]); n_val = int(n*split[1]); n_test = n - n_train - n_val
        parts = [("train",imgs[:n_train]),
                 ("val",imgs[n_train:n_train+n_val]),
                 ("test",imgs[n_train+n_val:])]
        for part_name, files in parts:
            dst = os.path.join(root, part_name, cname)
            os.makedirs(dst, exist_ok=True)
            for fp in files:
                shutil.copy2(fp, os.path.join(dst, os.path.basename(fp)))
    return train_dir, val_dir, test_dir

def get_loaders(data_dir: str, img_size: int, batch_size: int, workers: int,
                auto_split: bool, class_names: List[str],
                use_weighted_sampler: bool=False):
    if auto_split:
        train_root, val_root, test_root = auto_split_if_needed(data_dir, class_names)
    else:
        train_root = os.path.join(data_dir,"train")
        val_root   = os.path.join(data_dir,"val")
        test_root  = os.path.join(data_dir,"test")

    train_tf, val_tf = build_transforms(img_size)
    train_ds = datasets.ImageFolder(train_root, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_root, transform=val_tf)
    test_ds  = datasets.ImageFolder(test_root, transform=val_tf)

    if use_weighted_sampler:
        counts = Counter([y for _,y in train_ds.samples])
        total = sum(counts.values())
        class_weights = {cls_idx: total/count for cls_idx,count in counts.items()}
        sample_weights = [class_weights[y] for _,y in train_ds.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    val_loader  = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.class_to_idx
