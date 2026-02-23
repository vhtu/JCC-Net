import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer

from configs.config import *
from data.data_utils import load_all_data
from data.dataset import SimCLRDataset
from models.model import FullContrastiveModel
from models.losses import ContrastiveLosses
from validate import validate


def main():

    print("Loading dataset...")
    full_data = load_all_data(DATASET_ROOT, JSON_PATH)

    labels = [i['label'] for i in full_data]

    train_data, temp_data = train_test_split(
        full_data,
        test_size=0.4,
        stratify=labels,
        random_state=42
    )

    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        stratify=[i['label'] for i in temp_data],
        random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_loader = DataLoader(
        SimCLRDataset(train_data, tokenizer, IMG_SIZE, MAX_SEQ_LEN, True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        SimCLRDataset(val_data, tokenizer, IMG_SIZE, MAX_SEQ_LEN, False),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Initializing model...")
    model = FullContrastiveModel(NUM_CLASSES).to(DEVICE)

    contrastive_loss = ContrastiveLosses(TEMPERATURE)
    criterion_cls = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"\n--- TRAINING START ---")
    print(f"CLS={ALPHA_CLS} | CLIP={BETA_CLIP} | SIMCLR={GAMMA_SIMCLR}")

    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):

        model.train()
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")

        for batch in loop:

            v1 = batch['img_v1'].to(DEVICE)
            v2 = batch['img_v2'].to(DEVICE)
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)

            optimizer.zero_grad()

            logits, z_img1, z_img2, z_text = model(v1, v2, ids, mask)

            l_cls = criterion_cls(logits, lbl)
            l_clip = contrastive_loss.clip_loss(z_img1, z_text)
            l_simclr = contrastive_loss.simclr_loss(z_img1, z_img2)

            loss = (
                ALPHA_CLS * l_cls +
                BETA_CLIP * l_clip +
                GAMMA_SIMCLR * l_simclr
            )

            loss.backward()
            optimizer.step()

            loop.set_postfix(
                total=loss.item(),
                cls=l_cls.item(),
                clip=l_clip.item(),
                simclr=l_simclr.item()
            )

        # ==========================
        # VALIDATION
        # ==========================
        val_acc = validate(model, val_loader)

        print(f"Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("✔ Best model saved")

    print(f"\nTraining complete. Best Val Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()