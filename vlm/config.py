import torch

class Config:
    # Model
    VIT_MODEL_NAME = 'google/vit-base-patch16-224'
    GPT2_MODEL_NAME = 'gpt2'
    NUM_CROSS_ATTN_LAYERS = 6
    FREEZE_VISION = True
    FREEZE_LANGUAGE = True # Based on original script, might want to check this
    
    # Training
    LEARNING_RATE = 2e-5 # Based on typical fine-tuning
    NUM_EPOCHS = 5
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 1
    MAX_GRAD_NORM = 1.0
    SAVE_EVERY = 1
    
    # Checkpointing
    CHECKPOINT_DIR = 'checkpoints'
    LATEST_CHECKPOINT_NAME = 'latest_checkpoint.pth'
    
    # Generation
    MAX_LENGTH = 50
    TEMPERATURE = 0.7
    
    DATA_DIR = 'flickr8k' 
    IMAGE_DIR = 'flickr8k/train/Images'
    
    # System
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def setup_device(cls):
        print(f"🔧 Using device: {cls.DEVICE}")
        return torch.device(cls.DEVICE)
