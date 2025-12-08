import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import csv
from tqdm import tqdm

def get_transforms(split='train'):
    """
    Returns standard image transformations.
    Args:
        split (str): 'train' or 'val'/'test'. Training can have augmentation.
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

class ImageCaptioningDataset(Dataset):
    def __init__(self, root_dir, tokenizer, transform=None, max_length=50, split='train'):
        """
        Args:
            root_dir (str): Root directory containing 'Images' folder and 'captions.txt'. 
                            Example: ./flickr8k/train/
            tokenizer: GPT2Tokenizer instance.
            transform (callable, optional): Optional transform to be applied on a sample.
            max_length (int): Maximum length of the caption tokens.
            split (str): 'train', 'val', or 'test'. Used for transform selection/logging.
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'Images')
        self.caption_file = os.path.join(root_dir, 'captions.txt')
        
        self.tokenizer = tokenizer
        self.transform = transform if transform else get_transforms(split)
        self.max_length = max_length
        self.data = []

        print(f"Loading {split} data from {self.caption_file}...")
        
        if not os.path.exists(self.caption_file):
             print(f"Warning: Caption file {self.caption_file} not found. Dataset will be empty.")
             return

        try:
            with open(self.caption_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, skipinitialspace=True)
                next(reader, None) 
                
                for row in reader:
                    if len(row) < 2: 
                        continue
                    
                    image_name = row[0].strip()
                    caption_text = row[1].strip()
                    if caption_text.startswith("'") and caption_text.endswith("'"):
                        caption_text = caption_text[1:-1]
                    elif caption_text.startswith('"') and caption_text.endswith('"'):
                        caption_text = caption_text[1:-1]

                    self.data.append({
                        'image': image_name,
                        'caption': caption_text
                    })
        except Exception as e:
            print(f"Error reading caption file: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_name = item['image']
        caption_text = item['caption']

        img_path = os.path.join(self.image_dir, image_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros((3, 224, 224))
        
        # Tokenize Caption
        encodings = self.tokenizer(
            f"<|startoftext|>{caption_text}<|endoftext|>",
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'caption_text': caption_text
        }

def get_data_loader(dataset, batch_size=32, shuffle=True, num_workers=0):
    """Returns a DataLoader."""
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )
