import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from transformers import BertTokenizer

from configs.config import *
from data.dataset import CLIPDataset
from data.data_utils import load_all_data
from models.clip_model import CLIPMultimodalModel
from models.losses import CLIPLoss


def train_one_epoch(model, loader, optimizer, cls_loss_fn, clip_loss_fn):
    model.train()
    total_loss = 0

    loop = tqdm(loader, desc="Training")
    for batch in loop:
        imgs = batch['image'].to(DEVICE)
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()

        logits, z_img, z_txt = model(imgs, ids, mask)

        loss_cls = cls_loss_fn(logits, labels)
        loss_clip = clip_loss_fn(z_img, z_txt)

        loss = ALPHA_CLS * loss_cls + BETA_CLIP * loss_clip

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        loop.set_postfix(
            total_loss=loss.item(),
            cls=loss_cls.item(),
            clip=loss_clip.item()
        )

    return total_loss / len(loader)


def validate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(DEVICE)
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            logits, _, _ = model(imgs, ids, mask)
            preds = torch.argmax(logits, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    return acc


def main():
    print("Loading dataset...")
    full_data = load_all_data(DATASET_ROOT, JSON_PATH)

    labels = [item['label'] for item in full_data]

    train_data, temp_data = train_test_split(
        full_data, test_size=0.4,
        stratify=labels, random_state=42
    )

    val_data, test_data = train_test_split(
        temp_data, test_size=0.5,
        stratify=[i['label'] for i in temp_data],
        random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_loader = DataLoader(
        CLIPDataset(train_data, tokenizer, IMG_SIZE, MAX_SEQ_LEN, True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        CLIPDataset(val_data, tokenizer, IMG_SIZE, MAX_SEQ_LEN, False),
        batch_size=BATCH_SIZE
    )

    print("Initializing model...")
    model = CLIPMultimodalModel(NUM_CLASSES).to(DEVICE)

    cls_loss_fn = nn.CrossEntropyLoss()
    clip_loss_fn = CLIPLoss(TEMPERATURE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    print("\n===== TRAINING START =====")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

        train_loss = train_one_epoch(
            model, train_loader,
            optimizer,
            cls_loss_fn,
            clip_loss_fn
        )

        val_acc = validate(model, val_loader)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                "checkpoints/best_model_clip_only.pth"
            )
            print("✔ Saved best model")

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")
    print("Training completed.")


if __name__ == "__main__":
    main()