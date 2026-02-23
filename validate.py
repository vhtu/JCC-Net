import torch
from configs.config import DEVICE


def validate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in loader:

            v1 = batch['img_v1'].to(DEVICE)
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)

            logits, _, _, _ = model(v1, None, ids, mask)

            preds = torch.argmax(logits, dim=1)

            total += lbl.size(0)
            correct += (preds == lbl).sum().item()

    acc = 100 * correct / total

    return acc