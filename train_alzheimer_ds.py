import os
import time
import argparse
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import deepspeed
from deepspeed.accelerator import get_accelerator

##### Dataset (no torchvision) #####

CLASS_MAP_TRAIN = {
    "mild_dementia": 0,
    "moderated_dementia": 1,
    "non_demented": 2,
    "very_mild_dementia": 3,
}

CLASS_MAP_TEST = {
    "MildDemented": 0,
    "ModerateDemented": 1,
    "NonDemented": 2,
    "VeryMildDemented": 3,
}

class AlzheimerMRIDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", img_size: int = 224):
        if split == "train":
            base = os.path.join(root_dir, "train_images")
            class_map = CLASS_MAP_TRAIN
        else:
            base = os.path.join(root_dir, "test_images")
            class_map = CLASS_MAP_TEST

        self.samples: List[Tuple[str, int]] = []
        for cls_name, label in class_map.items():
            cls_dir = os.path.join(base, cls_name)
            if os.path.isdir(cls_dir):
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(cls_dir, fname), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {base}")

        self.img_size = img_size
        print(f"[{split}] Loaded {len(self.samples)} images from {base}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0  # HWC
        img_np = (img_np - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_np = np.transpose(img_np, (2, 0, 1))  # CHW
        return torch.from_numpy(img_np), torch.tensor(label, dtype=torch.long)

##### Simple CNN #####

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d((4,4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

##### Helpers #####

def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()

def prepare_dataloaders(root_dir, batch_size, num_workers, pin_memory):
    train_ds = AlzheimerMRIDataset(root_dir, split="train", img_size=224)
    test_ds  = AlzheimerMRIDataset(root_dir, split="test", img_size=224)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=2,
        persistent_workers=num_workers > 0, drop_last=True)
    val_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=max(1,num_workers//2), pin_memory=pin_memory, prefetch_factor=1,
        persistent_workers=False)
    return train_loader, val_loader

##### Main #####

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--mixed_precision", action="store_true", default=True)
    parser.add_argument("--local_rank", type=int, default=-1)  # Needed by deepspeed
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    get_accelerator().set_device(local_rank)
    device = get_accelerator().device_name(local_rank)

    train_loader, val_loader = prepare_dataloaders(
        args.data_dir, args.batch_size, args.num_workers, args.pin_memory
    )

    num_classes = 4
    model = SimpleCNN(num_classes)
    criterion = nn.CrossEntropyLoss()

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    target_dtype = torch.float16 if args.mixed_precision and model_engine.fp16_enabled() else torch.float32
    best_acc = 0.0

    for epoch in range(args.num_epochs):
        model_engine.train()
        running_loss, running_acc = 0.0, 0.0
        n_batches = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True, dtype=target_dtype)
            labels = labels.to(device, non_blocking=True)
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)
            model_engine.backward(loss)
            model_engine.step()
            acc = accuracy(outputs, labels)
            running_loss += loss.item()
            running_acc += acc
            n_batches += 1

        if local_rank == 0:
            print(f"[Epoch {epoch+1}] Train Loss: {running_loss/n_batches:.4f}  Train Acc: {running_acc/n_batches:.4f}")

        # Validation
        if local_rank == 0:
            model_engine.eval()
            val_loss, val_acc, val_batches = 0.0, 0.0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device, non_blocking=True, dtype=target_dtype)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model_engine(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_acc += accuracy(outputs, labels)
                    val_batches += 1
            print(f"[Epoch {epoch+1}] Val Loss: {val_loss/val_batches:.4f}  Val Acc: {val_acc/val_batches:.4f}")

if __name__ == "__main__":
    main()
