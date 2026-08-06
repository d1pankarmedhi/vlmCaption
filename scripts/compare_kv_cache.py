import sys
import os
import time
import torch
import torch.nn.functional as F
from PIL import Image

# Add parent directory to path to import vlm module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm.model import VisionLanguageModel
from vlm.data import get_transforms

@torch.no_grad()
def generate_caption_kv_cache(model, image, max_length=50, temperature=1.0, top_k=50):
    model.eval()
    device = next(model.parameters()).device

    # Prepare image
    if image.dim() == 3:
        image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Extract image features
    vision_outputs = model.vision_encoder(pixel_values=image)
    image_features = vision_outputs.last_hidden_state
    image_features = model.vision_projection(image_features)

    # Start with BOS token
    generated = torch.tensor(
        [[model.tokenizer.bos_token_id]],
        device=device
    )

    from transformers.cache_utils import DynamicCache
    past_key_values = DynamicCache()

    cross_attn_caches = [None] * len(model.cross_attention_layers)

    for _ in range(max_length):
        if past_key_values.get_seq_length() > 0:
            position_ids = torch.tensor([[generated.shape[1] - 1]], device=device)
            current_tokens = generated[:, -1:]
        else:
            position_ids = torch.arange(generated.shape[1], device=device).unsqueeze(0)
            current_tokens = generated

        text_embeds = model.language_decoder.transformer.wte(current_tokens)
        position_embeds = model.language_decoder.transformer.wpe(position_ids)
        hidden_states = text_embeds + position_embeds

        cross_attn_idx = 0
        for i, block in enumerate(model.language_decoder.transformer.h):
            # The past_key_values cache is updated in-place by the block
            outputs = block(hidden_states, past_key_values=past_key_values, use_cache=True)
            hidden_states = outputs[0]

            if i in model.cross_attn_positions and cross_attn_idx < len(model.cross_attention_layers):
                hidden_states, new_cache = model.cross_attention_layers[cross_attn_idx](
                    text_features=hidden_states,
                    image_features=image_features,
                    image_kv_cache=cross_attn_caches[cross_attn_idx]
                )
                cross_attn_caches[cross_attn_idx] = new_cache
                cross_attn_idx += 1

        hidden_states = model.language_decoder.transformer.ln_f(hidden_states)
        logits = model.language_decoder.lm_head(hidden_states)

        next_token_logits = logits[:, -1, :] / temperature

        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == model.tokenizer.eos_token_id:
            break

        generated = torch.cat([generated, next_token], dim=1)

    caption = model.tokenizer.decode(generated[0], skip_special_tokens=True)
    return caption

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    model = VisionLanguageModel()
    
    # Try to load checkpoint if it exists
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "latest_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print("Found checkpoint, loading weights...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No checkpoint found. Using untrained weights.")
    
    model.to(device)
    model.eval()

    image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "image.png")
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    transform = get_transforms()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).to(device)

    # Warmup
    print("Warming up models...")
    _ = model.generate_caption(image_tensor, max_length=20)
    _ = generate_caption_kv_cache(model, image_tensor, max_length=20)
    
    # We force deterministic generation by greedy decoding for a fair comparison of identical output lengths
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("\n--- Testing Without KV Caching ---")
    start_time = time.time()
    with torch.no_grad():
        caption_no_kv = model.generate_caption(image_tensor, max_length=50, top_k=1)
    end_time = time.time()
    time_no_kv = end_time - start_time
    print(f"Caption: {caption_no_kv}")
    print(f"Time taken: {time_no_kv:.4f} seconds")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("\n--- Testing With KV Caching ---")
    start_time = time.time()
    with torch.no_grad():
        caption_kv = generate_caption_kv_cache(model, image_tensor, max_length=50, top_k=1)
    end_time = time.time()
    time_kv = end_time - start_time
    print(f"Caption: {caption_kv}")
    print(f"Time taken: {time_kv:.4f} seconds")

    print("\n--- Results ---")
    print(f"Without KV Cache: {time_no_kv:.4f} s")
    print(f"With KV Cache:    {time_kv:.4f} s")
    if time_no_kv > 0:
        speedup = time_no_kv / time_kv
        print(f"Speedup:          {speedup:.2f}x")

    assert caption_no_kv == caption_kv, "Mismatch in generated captions! KV caching implementation might have an issue."

if __name__ == "__main__":
    main()
