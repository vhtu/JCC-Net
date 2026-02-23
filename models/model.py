import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.net(x)


class FullContrastiveModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.cnn = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        self.cnn.classifier = nn.Identity()
        self.cnn.avgpool = nn.Identity()

        self.img_proj = Projector(1280, 512)

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for p in self.bert.parameters():
            p.requires_grad = False

        self.text_proj = Projector(768, 512)

        self.cross_att = nn.MultiheadAttention(512, 4, batch_first=True)
        self.norm = nn.LayerNorm(512)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, n_classes)
        )

    def encode_image(self, img):
        feat = self.cnn.features(img)
        b, c, h, w = feat.shape

        seq = feat.view(b, c, h*w).permute(0, 2, 1)
        emb_seq = self.img_proj(seq)
        emb_global = emb_seq.mean(dim=1)

        return emb_seq, emb_global

    def forward(self, img_v1, img_v2, input_ids, mask):

        img_seq_v1, img_global_v1 = self.encode_image(img_v1)

        img_global_v2 = None
        if img_v2 is not None:
            _, img_global_v2 = self.encode_image(img_v2)

        txt_out = self.bert(input_ids, mask)
        txt_seq = self.text_proj(txt_out.last_hidden_state)
        txt_global = txt_seq[:, 0, :]

        att_out, _ = self.cross_att(
            query=img_seq_v1,
            key=txt_seq,
            value=txt_seq
        )

        fused = self.norm(img_seq_v1 + att_out).mean(dim=1)
        logits = self.fc(fused)

        return logits, img_global_v1, img_global_v2, txt_global