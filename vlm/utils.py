import torch
import numpy as np
import random
import os
from nltk.translate.bleu_score import corpus_bleu

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_bleu_scores(references, hypotheses):
    """
    Calculate BLEU scores (1-4).
    
    Args:
        references: List of list of reference tokens (ground truth).
                    Example: [[['hello', 'world']], [['cat', 'sat']]]
        hypotheses: List of hypothesis tokens (generated).
                    Example: [['hello', 'world'], ['cat', 'sitting']]
    
    Returns:
        Dictionary containing BLEU-1 to BLEU-4 scores.
    """
    # BLEU-N measures n-gram precision
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    return {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-3': bleu3,
        'BLEU-4': bleu4
    }

def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state, filename):
    """Save training checkpoint."""
    print(f"💾 Saving checkpoint to {filename}")
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer=None, scaler=None, scheduler=None, device='cpu'):
    """Load training checkpoint."""
    if not os.path.exists(filename):
        print(f"📂 No checkpoint found at {filename}")
        return None
        
    print(f"📂 Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location=device)
    
    if model:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    return checkpoint
