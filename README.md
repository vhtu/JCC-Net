# JCC-Net: Joint Cross-Modal Contrastive Network for Multimodal Fish and Shrimp Disease Classification

## 1. Abstract

This repository presents **JCC-Net**, a robust multimodal deep learning framework that integrates visual imagery with symptom descriptions to enhance diagnostic precision.

------------------------------------------------------------------------

## 2. Dataset

The Fish & Shrimp Disease Dataset used in this study is publicly
available on Kaggle:

🔗 Dataset Link:
https://www.kaggle.com/datasets/vohoangtu/fish-shrimp-db-new/data

Dataset directory structure:

    Dataset_Fish_Shrimp_New/
    │
    ├── train/
    ├── valid/
    ├── test/
    └── fish_shrimp_dataset_openai.json

Each JSON entry contains:

``` json
{
  "file_name": "image_001.jpg",
  "caption": "Microscopic view of infected shrimp tissue"
}
```

------------------------------------------------------------------------


## 3. Project Structure

    JCC-Net/
    │
    ├── configs/
    │   └── config.py
    │
    ├── data/
    │   ├── dataset.py
    │   └── data_utils.py
    │
    ├── models/
    │   ├── clip_model.py
    │   ├── projector.py
    │   └── losses.py
    │
    ├── checkpoints/
    ├── results/
    │
    ├── train.py
    ├── evaluate.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------



## 4. Installation

``` bash
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

------------------------------------------------------------------------

## 5. Training

``` bash
python train.py
```

Best model saved at:

    checkpoints/best_model_clip_only.pth

------------------------------------------------------------------------

## 6. Evaluation

``` bash
python evaluate.py
```

Outputs:

-   Classification report
-   Confusion matrix
-   results/confusion_matrix_clip_only.png


# JCC-Net
