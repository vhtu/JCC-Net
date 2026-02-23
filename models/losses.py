import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLosses(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature
        self.ce = nn.CrossEntropyLoss()

    def clip_loss(self, img_emb, text_emb):
        img_emb = F.normalize(img_emb, dim=1)
        text_emb = F.normalize(text_emb, dim=1)

        logits = (img_emb @ text_emb.T) / self.temp
        labels = torch.arange(img_emb.shape[0], device=img_emb.device)

        return (self.ce(logits, labels) +
                self.ce(logits.T, labels)) / 2

    def simclr_loss(self, z_i, z_j):
        batch_size = z_i.shape[0]

        features = torch.cat([z_i, z_j], dim=0)
        features = F.normalize(features, dim=1)

        logits = torch.matmul(features, features.T) / self.temp
        logits.fill_diagonal_(-float('inf'))

        labels = torch.cat([
            torch.arange(batch_size, device=z_i.device) + batch_size,
            torch.arange(batch_size, device=z_i.device)
        ], dim=0)

        return self.ce(logits, labels)