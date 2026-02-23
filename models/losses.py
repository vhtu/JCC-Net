import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, img_emb, text_emb):
        img_emb = F.normalize(img_emb, dim=1)
        text_emb = F.normalize(text_emb, dim=1)

        logits = (img_emb @ text_emb.T) / self.temp
        labels = torch.arange(img_emb.size(0), device=img_emb.device)

        return (self.ce(logits, labels) +
                self.ce(logits.T, labels)) / 2