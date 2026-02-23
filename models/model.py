import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertModel


# ======================================================
# Projection Head
# ======================================================
class Projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ======================================================
# JCC-Net v2 (SimCLR + CLIP + Cross Attention)
# ======================================================
class JCCNet(nn.Module):
    def __init__(self, n_classes, temperature=0.07):
        super().__init__()

        self.temperature = temperature

        # ==============================
        # 1️⃣ Image Encoder
        # ==============================
        self.cnn = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        self.cnn.classifier = nn.Identity()

        self.img_proj = Projector(1280, 512)

        # ==============================
        # 2️⃣ Text Encoder
        # ==============================
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for p in self.bert.parameters():
            p.requires_grad = False

        self.txt_proj = Projector(768, 512)

        # ==============================
        # 3️⃣ Cross Modal Attention
        # ==============================
        self.cross_att = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=4,
            batch_first=True
        )

        self.norm = nn.LayerNorm(512)

        # ==============================
        # 4️⃣ Classifier
        # ==============================
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, n_classes)
        )

    # ==================================================
    # SimCLR Loss (Image-Image)
    # ==================================================
    def simclr_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        batch_size = z1.size(0)

        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T)

        labels = torch.arange(batch_size).to(z1.device)
        labels = torch.cat([labels, labels], dim=0)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z1.device)
        similarity_matrix = similarity_matrix / self.temperature
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


    # ==================================================
    # CLIP Loss (Image-Text)
    # ==================================================
    def clip_loss(self, img_emb, txt_emb):

        img_emb = F.normalize(img_emb, dim=1)
        txt_emb = F.normalize(txt_emb, dim=1)

        logits = torch.matmul(img_emb, txt_emb.T) / self.temperature
        labels = torch.arange(img_emb.size(0)).to(img_emb.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)

        return (loss_i + loss_t) / 2


    # ==================================================
    # Forward
    # ==================================================
    def forward(self, img_view1, img_view2, input_ids, attention_mask):

        # ==================================================
        # 1️⃣ IMAGE BRANCH (Two Views)
        # ==================================================
        feat1 = self.cnn.features(img_view1)
        feat2 = self.cnn.features(img_view2)

        b, c, h, w = feat1.shape

        seq1 = feat1.view(b, c, h*w).permute(0, 2, 1)
        seq2 = feat2.view(b, c, h*w).permute(0, 2, 1)

        emb_seq1 = self.img_proj(seq1)
        emb_seq2 = self.img_proj(seq2)

        img_global1 = emb_seq1.mean(dim=1)
        img_global2 = emb_seq2.mean(dim=1)

        # ==================================================
        # 2️⃣ TEXT BRANCH
        # ==================================================
        txt_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        txt_seq = self.txt_proj(txt_out.last_hidden_state)
        txt_global = txt_seq[:, 0, :]

        # ==================================================
        # 3️⃣ Cross Modal Fusion (View1 used)
        # ==================================================
        att_out, _ = self.cross_att(
            query=emb_seq1,
            key=txt_seq,
            value=txt_seq
        )

        fused = self.norm(emb_seq1 + att_out).mean(dim=1)
        logits = self.classifier(fused)

        # ==================================================
        # 4️⃣ Loss Components
        # ==================================================
        simclr = self.simclr_loss(img_global1, img_global2)
        clip = self.clip_loss(img_global1, txt_global)

        return logits, simclr, clip