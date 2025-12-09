<div align="center">

# VLM Captioning

A Visual Language Model for Image Captioning using ViT (Vision Transformer) and GPT-2.

<img width="400" alt="image" src="https://github.com/user-attachments/assets/f0c3d285-5ca8-4313-841a-717382ef4800" />
<p>Fig: Model Architecture</p>

</div>

## Getting Started

### Setup with uv

1.  **Initialize Project**:
    ```bash
    uv init
    ```

2.  **Create Virtual Environment**:
    ```bash
    uv venv
    ```
    Activate it:
    *   Windows: `.venv\Scripts\activate`
    *   Linux/macOS: `source .venv/bin/activate`

3.  **Install Requirements**:
    ```bash
    uv pip install -r requirements.txt
    # OR if using pyproject.toml
    uv sync
    ```

### Data Preparation

This model has been trained on `flickr8k` dataset that has 8,000 image-text pairs, with 5 captions per image. You can download it from [here](https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip).

```
# download and unzip flickr8k dataset
!wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip"
!unzip -q flickr8k.zip -d ./flickr8k
!rm flickr8k.zip
!echo "Downloaded Flickr8k dataset successfully."
```

Downloaded dataset will be in the following format:
```
flickr8k/
    Images/
        1.jpg
        2.jpg
        ...
    captions.txt
```

You have to split the dataset into `train`, `test` and `val` sets. The `captions.txt` file contains the image-text pairs. The first column is the image name and the second column is the caption. The dataset, after splitting, should be in the following format:
```
flickr8k/
    train/
        Images/
            1.jpg
            2.jpg
            ...
        captions.txt
    val/
        Images/
            1.jpg
            2.jpg
            ...
        captions.txt
    test/
        Images/
            1.jpg
            2.jpg
            ...
        captions.txt
```


### Training

```bash
python main.py train --dataset_dir flickr8k/ --epochs 10 --batch_size 16
```

### Inference

Generate caption for a single image:
```bash
python main.py infer --image_path data/image.jpg --checkpoint_path checkpoints/latest_checkpoint.pth
```
<table>
<tr>
    <td><img src="data\image.png" width="300"></td>
    <td><b>Generated Caption:</b> The golden retriever is carrying a yellow ball in its mouth as he bounds towards it .</td>
</tr>
</table>

> Download the latest checkpoint from releases: [latest_checkpoint.pth](https://github.com/d1pankarmedhi/vlmCaption/releases/tag/v1.0.0)