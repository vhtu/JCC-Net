import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer

from configs.config import *
from data.dataset import CLIPDataset
from data.data_utils import load_all_data
from models.model import JCCNet


def evaluate():
    print("Loading dataset...")
    full_data = load_all_data(DATASET_ROOT, JSON_PATH)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    test_loader = DataLoader(
        CLIPDataset(full_data, tokenizer, IMG_SIZE, MAX_SEQ_LEN, False),
        batch_size=BATCH_SIZE
    )

    model = JCCNet(NUM_CLASSES).to(DEVICE)

    weight_path = "checkpoints/best_model_clip_only.pth"

    if not os.path.exists(weight_path):
        print("Weight file not found!")
        return

    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    print("✔ Loaded best model")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].to(DEVICE)
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            logits, _, _ = model(imgs, ids, mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    target_names = [f"Class {i}" for i in range(NUM_CLASSES)]

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        digits=4
    ))

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
    plt.title("Confusion Matrix (CLIP Only)")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix_clip_only.png")
    print("✔ Saved confusion matrix")


if __name__ == "__main__":
    evaluate()