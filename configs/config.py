import os
import torch

# =========================
# DATASET
# =========================
DATASET_ROOT = 'fish-shrimp-db-new/Dataset_Fish_Shrimp_New'
JSON_PATH = os.path.join(DATASET_ROOT, 'fish_shrimp_dataset_openai.json')

NUM_CLASSES = 11
IMG_SIZE = (224, 224)
MAX_SEQ_LEN = 64

# =========================
# TRAINING
# =========================
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOSS WEIGHTS
# =========================
ALPHA_CLS = 0.6
BETA_CLIP = 0.2
GAMMA_SIMCLR = 0.2
TEMPERATURE = 0.07

# =========================
# SAVE PATH
# =========================
CHECKPOINT_PATH = "best_model_full_contrastive.pth"