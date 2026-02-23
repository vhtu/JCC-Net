import os
import json

def clean_caption_text(text):
    text = text.lower()
    banned = ['healthy', 'white spot', 'disease', 'symptom', 'infected', 'spots', 'wssv', 'bg']
    for w in banned:
        text = text.replace(w, '[MASK]')
    return text


def load_all_data(root_dir, json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_json = json.load(f)

    caption_lookup = {
        os.path.basename(item['file_name']): clean_caption_text(item['caption'])
        for item in raw_json
    }

    all_data = []
    for sub in ['train', 'valid', 'test']:
        sub_path = os.path.join(root_dir, sub)
        if not os.path.exists(sub_path):
            continue

        for cls in os.listdir(sub_path):
            try:
                class_id = int(cls)
            except:
                continue

            cls_path = os.path.join(sub_path, cls)
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_data.append({
                        'img_path': os.path.join(cls_path, img_name),
                        'caption': caption_lookup.get(img_name, "microscopic view"),
                        'label': class_id
                    })
    return all_data