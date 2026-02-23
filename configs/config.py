import torch
import os

DATASET_ROOT = 'Dataset_Fish_Shrimp_New'
JSON_PATH = os.path.join(DATASET_ROOT, 'fish_shrimp_dataset_openai.json')

BATCH_SIZE = 16
IMG_SIZE = (224, 224)
MAX_SEQ_LEN = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.01
NUM_CLASSES = 11

ALPHA_CLS = 0.7
BETA_CLIP = 0.3
TEMPERATURE = 0.07

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")