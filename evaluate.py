import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer

from configs.config import *
from data.data_utils import load_all_data
from data.dataset import SimCLRDataset
from models.model import FullContrastiveModel


def main():

    print("\n--- TESTING PHASE ---")

    # =========================
    # Load dataset
    # =========================
    full_data = load_all_data(DATASET_ROOT, JSON_PATH)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    test_loader = DataLoader(
        SimCLRDataset(full_data, tokenizer, IMG_SIZE, MAX_SEQ_LEN, False),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # =========================
    # Load model
    # =========================
    model = FullContrastiveModel(NUM_CLASSES).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"✔ Loaded weight from: {CHECKPOINT_PATH}")
    else:
        print(f"❌ Không tìm thấy {CHECKPOINT_PATH}")
        return

    model.eval()

    all_preds = []
    all_labels = []

    # =========================
    # Testing loop
    # =========================
    with torch.no_grad():

        for batch in test_loader:

            v1 = batch['img_v1'].to(DEVICE)
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)

            logits, _, _, _ = model(v1, None, ids, mask)

            _, predicted = torch.max(logits, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())

    # =========================
    # Metrics
    # =========================
    target_names = [f"Class {i}" for i in range(NUM_CLASSES)]

    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        digits=4
    ))

    # =========================
    # Confusion Matrix
    # =========================
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Full Contrastive Learning)')
    plt.savefig('confusion_matrix_full_contrastive.png')

    print("✔ Đã lưu confusion matrix.")


if __name__ == "__main__":
    main()