import torch
from tqdm import tqdm
from .utils import save_checkpoint, compute_bleu_scores

def train_epoch(model, dataloader, optimizer, device, epoch, scaler, grad_accum_steps=1, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # AMP Context
        with torch.amp.autocast(enabled=True): 
            outputs = model(images, input_ids, attention_mask)
            loss = outputs.loss / grad_accum_steps
            
        # Backward
        scaler.scale(loss).backward()
        
        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item() * grad_accum_steps
        progress_bar.set_postfix({'loss': loss.item() * grad_accum_steps})
        
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, tokenizer, device):
    """
    Evaluate model using BLEU scores.
    Args:
        model: VLM
        dataloader: Validation dataloader
        tokenizer: GPT2 Tokenizer
        device: torch device
    """
    model.eval()
    references = []
    hypotheses = []
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in progress_bar:
            images = batch['image'].to(device)            
            # Generate
            bs = images.size(0)

            # Encode images once for the batch
            encoder_outputs = model.encoder(images)
            image_features = encoder_outputs.last_hidden_state

            # Generate
            start_token = torch.full((bs, 1), tokenizer.bos_token_id, device=device, dtype=torch.long)
            generated_ids = model.decoder.generate(
                start_token,
                max_length=50,
                encoder_hidden_states=image_features,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            current_batch_refs = batch['caption_text']
            
            for i in range(bs):
                pred = decoded_preds[i].lower().split()
                ref = current_batch_refs[i].lower().split()
                
                hypotheses.append(pred)
                references.append([ref])
                
    return compute_bleu_scores(references, hypotheses)

