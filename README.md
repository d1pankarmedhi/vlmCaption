<div align="center">

# 🖼️ VLM Captioning

A fast and efficient Visual Language Model for Image Captioning combining **ViT (Vision Transformer)** for image encoding and **GPT-2** for text generation.

</div>

## 📖 Table of Contents
- [Overview](#-overview)
- [How the Model Works](#-how-the-model-works)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Usage](#-usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Helper Scripts](#-helper-scripts)
- [Checkpoints](#-checkpoints)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

**VLM Captioning** bridges the gap between vision and language by generating highly accurate captions for given images. It utilizes a state-of-the-art encoder-decoder architecture:
- **Vision Transformer (ViT)**: Extracts rich, high-level image features.
- **GPT-2**: Decodes the visual features into coherent and contextually relevant natural language.

---

## 🧠 How the Model Works

The model combines a pre-trained **Vision Transformer (ViT)** image encoder with a pre-trained **GPT-2** language decoder, connected via **interleaved Multi-Head Cross-Attention layers**.

### Architecture Overview Diagram


<img width="600" alt="Model Architecture" src="https://github.com/user-attachments/assets/f0c3d285-5ca8-4313-841a-717382ef4800" />
<br>
<em>Fig: ViT + GPT-2 Architecture</em>


### Component Details

1. **Vision Encoder (`ViTModel`)**:
   - Resizes input images to `224x224x3` and extracts 196 patch embeddings (`16x16` patches) plus 1 `[CLS]` token embedding.
   - Backbone parameters remain **frozen** to leverage pre-trained visual feature representations without catastrophic forgetting.

2. **Linear Vision Projection**:
   - Projects ViT feature dimensions (`768d`) to align with GPT-2's embedding dimension (`768d`).

3. **Language Decoder (`GPT2LMHeadModel`)**:
   - Pre-trained GPT-2 architecture (`12` Transformer layers, `768` hidden size).
   - Generates text token-by-token autoregressively.

4. **Interleaved Cross-Attention Layers (`CrossAttentionBlock`)**:
   - **6 trainable cross-attention blocks** are inserted at regular intervals across GPT-2's 12 transformer layers.
   - **Queries ($Q$)** are derived from text hidden states inside GPT-2.
   - **Keys ($K$)** and **Values ($V$)** are derived from the projected visual patch embeddings.
   - Allows language tokens to dynamically query and condition on relevant visual regions in the image.

5. **Training & Inference Strategy**:
   - **Training**: Computes Cross-Entropy loss on target token sequences using mixed precision (`torch.amp.autocast`). Only the cross-attention blocks, linear projection layer, and special token embeddings are trained.
   - **Inference**: Autoregressively generates tokens using top-k and temperature sampling starting from `<|startoftext|>` until `<|endoftext|>` is predicted.

---

## 🚀 Installation

We recommend using [`uv`](https://github.com/astral-sh/uv) for lightning-fast dependency management.

1. **Clone the repository** (optional if already downloaded):
   ```bash
   git clone https://github.com/d1pankarmedhi/vlmCaption.git
   cd vlmCaption
   ```

2. **Initialize Project**:
   ```bash
   uv init
   ```

3. **Create Virtual Environment**:
   ```bash
   uv venv
   ```
   **Activate it:**
   - **Windows**: `.venv\Scripts\activate`
   - **Linux/macOS**: `source .venv/bin/activate`

4. **Install Requirements**:
   ```bash
   uv pip install -r requirements.txt
   # OR if using pyproject.toml
   uv sync
   ```

---

## 📊 Data Preparation

The model is trained on the **Flickr8k** dataset, containing 8,000 image-text pairs (5 captions per image). 

### Automated Preparation (Recommended)

Run the automated helper script to download, unzip, and organize the dataset into `train`, `val`, and `test` splits:

```bash
bash scripts/download_data.sh flickr8k
```

### Manual Preparation

Alternatively, download and extract the dataset manually:

```bash
wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip"
unzip -q flickr8k.zip -d ./flickr8k
rm flickr8k.zip
```

**Dataset Structure:** 
Split the dataset into `train`, `val`, and `test` sets. The `captions.txt` file contains the image-text pairs (column 1: image name, column 2: caption). After splitting, your `flickr8k` folder should look like this:

```text
flickr8k/
├── train/
│   ├── Images/
│   └── captions.txt
├── val/
│   ├── Images/
│   └── captions.txt
└── test/
    ├── Images/
    └── captions.txt
```

---

## 💻 Usage

### Training

To train the model from scratch:

**Using Helper Script (Recommended):**
```bash
bash scripts/train.sh --dataset_dir flickr8k --epochs 10 --batch_size 16
```

**Using Python:**
```bash
python main.py train \
    --dataset_dir flickr8k/ \
    --epochs 10 \
    --batch_size 16
```

### Inference

Generate a caption for a single image using a trained checkpoint:

**Using Helper Script (Recommended):**
```bash
bash scripts/inference.sh data/image.jpg --checkpoint_path checkpoints/latest_checkpoint.pth
```

**Using Python:**
```bash
python main.py infer \
    --image_path data/image.jpg \
    --checkpoint_path checkpoints/latest_checkpoint.pth
```

#### Example Result:

<table>
  <tr>
    <td align="center"><img src="data/image.png" width="300" style="border-radius: 8px;"></td>
    <td align="center"><b>✨ Generated Caption:</b><br><br><i>"The golden retriever is carrying a yellow ball in its mouth as he bounds towards it."</i></td>
  </tr>
</table>

### 🛠️ Helper Scripts

The repository includes convenient shell scripts in the [`scripts/`](scripts/) directory for pipeline operations:

| Script | Description | Usage Example |
| :--- | :--- | :--- |
| [`scripts/download_data.sh`](scripts/download_data.sh) | Downloads Flickr8k and generates stratified train/val/test splits. | `bash scripts/download_data.sh [OUTPUT_DIR]` |
| [`scripts/train.sh`](scripts/train.sh) | Launches training via `uv run` with custom parameters (`--epochs`, `--batch_size`, `--lr`, `--mixed_precision`). | `bash scripts/train.sh --dataset_dir flickr8k --epochs 10` |
| [`scripts/inference.sh`](scripts/inference.sh) | Runs model inference on a target image with a specified checkpoint. | `bash scripts/inference.sh data/image.jpg -c checkpoints/latest.pth` |

---

## 📦 Checkpoints

You can download the weights from the releases page to run inference without training.

> 🔗 **Download:** [latest_checkpoint.pth (v1.0.0)](https://github.com/d1pankarmedhi/vlmCaption/releases/tag/v1.0.0)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.