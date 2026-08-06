#!/usr/bin/env bash
set -e

OUTPUT_DIR="${1:-flickr8k}"
URL="https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip"
ZIP_FILE="flickr8k.zip"
TEMP_DIR="flickr8k_temp"

echo "=========================================="
echo " Downloading Flickr8k Dataset"
echo "=========================================="

if [ -d "$OUTPUT_DIR/train" ] && [ -d "$OUTPUT_DIR/val" ]; then
  echo "Dataset directory '$OUTPUT_DIR' already contains train and val splits."
  echo "Skipping download."
  exit 0
fi

# Step 1: Download
if [ ! -f "$ZIP_FILE" ]; then
  echo "Downloading dataset from $URL..."
  if command -v curl &> /dev/null; then
    curl -L -o "$ZIP_FILE" "$URL"
  elif command -v wget &> /dev/null; then
    wget -O "$ZIP_FILE" "$URL"
  else
    echo "Error: Neither curl nor wget is installed."
    exit 1
  fi
else
  echo "Found existing $ZIP_FILE, skipping download."
fi

# Step 2: Unzip
echo "Extracting $ZIP_FILE..."
mkdir -p "$TEMP_DIR"
unzip -q -o "$ZIP_FILE" -d "$TEMP_DIR"

# Step 3: Organize & Split Data using Python
echo "Organizing dataset into train/val/test splits..."
python - <<EOF
import os
import csv
import shutil
import random

temp_dir = "$TEMP_DIR"
output_dir = "$OUTPUT_DIR"

caption_file = None
image_dir = None

for root, dirs, files in os.walk(temp_dir):
    if "captions.txt" in files:
        caption_file = os.path.join(root, "captions.txt")
    for d in dirs:
        if d.lower() == "images":
            image_dir = os.path.join(root, d)
            break

if not caption_file:
    raise FileNotFoundError(f"Could not find captions.txt in {temp_dir}")
if not image_dir:
    raise FileNotFoundError(f"Could not find Images directory in {temp_dir}")

print(f"Found captions at: {caption_file}")
print(f"Found images at: {image_dir}")

captions_by_image = {}
with open(caption_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, skipinitialspace=True)
    next(reader, None)
    for row in reader:
        if len(row) >= 2:
            img = row[0].strip()
            cap = row[1].strip()
            if img not in captions_by_image:
                captions_by_image[img] = []
            captions_by_image[img].append(cap)

image_list = sorted(list(captions_by_image.keys()))
random.seed(42)
random.shuffle(image_list)

n = len(image_list)
n_train = int(n * 0.8)
n_val = int(n * 0.1)

splits = {
    'train': image_list[:n_train],
    'val': image_list[n_train:n_train + n_val],
    'test': image_list[n_train + n_val:]
}

for split, img_names in splits.items():
    split_dir = os.path.join(output_dir, split)
    split_img_dir = os.path.join(split_dir, "Images")
    os.makedirs(split_img_dir, exist_ok=True)
    
    split_caption_path = os.path.join(split_dir, "captions.txt")
    with open(split_caption_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "caption"])
        for img in img_names:
            src_img = os.path.join(image_dir, img)
            dst_img = os.path.join(split_img_dir, img)
            if os.path.exists(src_img) and not os.path.exists(dst_img):
                shutil.move(src_img, dst_img)
            for cap in captions_by_image[img]:
                writer.writerow([img, cap])
                
    print(f"Created {split} split with {len(img_names)} images in {split_dir}")

EOF

# Step 4: Cleanup
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"
rm -f "$ZIP_FILE"

echo "Dataset successfully prepared in ./$OUTPUT_DIR!"
