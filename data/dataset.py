import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class SimCLRDataset(Dataset):
    def __init__(self, data_list, tokenizer, img_size, max_len, is_train=False):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        try:
            img = cv2.imread(item['img_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.is_train:
            img_v1 = self.train_transform(img)
            img_v2 = self.train_transform(img)
        else:
            img_v1 = self.val_transform(img)
            img_v2 = img_v1

        encoded = self.tokenizer.encode_plus(
            item['caption'],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'img_v1': img_v1,
            'img_v2': img_v2,
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }