import argparse
import torch
import os
from transformers import GPT2Tokenizer
from torch.amp import GradScaler

from vlm.config import Config
from vlm.model import VisionLanguageModel
from vlm.data import ImageCaptioningDataset, get_data_loader
from vlm.utils import set_seed, count_parameters, save_checkpoint, load_checkpoint
from vlm.train import train_epoch, evaluate_model
from vlm.inference import generate_caption

def train_command(args):
    """
    Handle training command.
    """
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(Config.GPT2_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Datasets
    print("\nLoading Datasets...")
    dataset_dir = args.dataset_dir
    
    # Train Set
    train_dir = os.path.join(dataset_dir, 'train')
    train_dataset = ImageCaptioningDataset(train_dir, tokenizer, max_length=Config.MAX_LENGTH, split='train')
    
    # Val Set
    val_dir = os.path.join(dataset_dir, 'val')
    if not os.path.exists(val_dir):
        print(f"\nValidation directory not found at {val_dir}. Evaluation will be skipped.")
        val_dataset = None
    else:
        val_dataset = ImageCaptioningDataset(val_dir, tokenizer, max_length=Config.MAX_LENGTH, split='val')

    train_loader = get_data_loader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = get_data_loader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if val_dataset else None
    
    print(f"Train Size: {len(train_dataset)}")
    if val_dataset:
        print(f"Val Size: {len(val_dataset)}")

    # Model
    print("\nInitializing Model...")
    model = VisionLanguageModel(
        vit_model_name=Config.VIT_MODEL_NAME,
        gpt2_model_name=Config.GPT2_MODEL_NAME,
        num_cross_attn_layers=Config.NUM_CROSS_ATTN_LAYERS,
        freeze_vision=Config.FREEZE_VISION,
        freeze_language=Config.FREEZE_LANGUAGE
    )
    model = model.to(device)
    print(f"\nModel Parameters: {count_parameters(model):,}")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    # Resume training from checkpoint
    start_epoch = 1
    if args.resume:
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, Config.LATEST_CHECKPOINT_NAME)
        if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
             ckpt_path = args.checkpoint_path
             
        checkpoint = load_checkpoint(ckpt_path, model, optimizer, scaler, scheduler, device)
        if checkpoint:
             start_epoch = checkpoint['epoch'] + 1
             print(f"Resuming from epoch {start_epoch}")

    if not os.path.exists(Config.CHECKPOINT_DIR):
        os.makedirs(Config.CHECKPOINT_DIR)

    for epoch in range(start_epoch, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, epoch, scaler, Config.GRAD_ACCUM_STEPS, Config.MAX_GRAD_NORM)
        print(f"Epoch {epoch} Loss: {loss:.4f}")
        
        # Evaluate on validation set
        if val_loader and epoch % 5 == 0:
            metrics = evaluate_model(model, val_loader, tokenizer, device)
            print(f"Epoch {epoch} Metrics: {metrics}")
        
        # Save model checkpoint
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        save_path = os.path.join(Config.CHECKPOINT_DIR, Config.LATEST_CHECKPOINT_NAME)
        save_checkpoint(save_dict, save_path)
        
        # Also save numbered checkpoint
        epoch_path = os.path.join(Config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
        if epoch % Config.SAVE_EVERY == 0:
             save_checkpoint(save_dict, epoch_path)
        
        if scheduler:
            scheduler.step()

def infer_command(args):
    """
    Handle inference command.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    print("\nLoading Model...")
    model = VisionLanguageModel()
    
    # Load weights
    ckpt_path = args.checkpoint_path if args.checkpoint_path else os.path.join(Config.CHECKPOINT_DIR, Config.LATEST_CHECKPOINT_NAME)
    load_checkpoint(ckpt_path, model, device=device)
    model = model.to(device)
    
    print(f"\nGenerating caption for {args.image_path}...")
    caption = generate_caption(model, args.image_path, device)
    
    print(f"\nCaption: {caption}\n")


def main():
    parser = argparse.ArgumentParser(description="VLM Captioning CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Train Parser
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--dataset_dir', type=str, required=True, help='Root directory of dataset (must contain train/val folders)')
    train_parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS)
    train_parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    train_parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    train_parser.add_argument('--seed', type=int, default=Config.SEED)
    train_parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')
    train_parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    train_parser.add_argument('--checkpoint_path', type=str, help='Specific checkpoint to resume from')

    # Infer Parser
    infer_parser = subparsers.add_parser('infer', help='Generate caption for an image')
    infer_parser.add_argument('--image_path', type=str, required=True, help='Path to image file')
    infer_parser.add_argument('--checkpoint_path', type=str, help='Path to model checkpoint')

    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'infer':
        infer_command(args)

if __name__ == '__main__':
    main()
