import torch
import torch.nn as nn
from transformers import ViTModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention Layer that allows text tokens to attend to visual features.
    Args:
        hidden_dim: Dimension of both text and image embeddings (must match)
        num_heads: Number of attention heads for multi-head attention
        dropout: Dropout probability for regularization
    """
    def __init__(self, hidden_dim, num_heads=12, dropout=0.1):
        super().__init__()
        # Multi-head cross-attention: queries from text, keys/values from image
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  
        )

        # Layer normalization for stable training
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network for additional expressiveness
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(), # (used in GPT-2)
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, text_features, image_features, attention_mask=None):
        """
        Args:
            text_features: Text embeddings from GPT-2 (batch, text_seq_len, hidden_dim)
            image_features: Image embeddings from ViT (batch, num_patches, hidden_dim)
            attention_mask: Optional mask for padding tokens

        Returns:
            Enhanced text features after attending to image (batch, text_seq_len, hidden_dim)
        """
        # Cross-attention: text queries attend to image keys/values
        attn_output, _ = self.cross_attn(
            query=text_features,
            key=image_features,
            value=image_features,
            key_padding_mask=attention_mask
        )

        # Residual connection + Layer norm
        text_features = self.layer_norm1(text_features + attn_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(text_features)
        text_features = self.layer_norm2(text_features + ffn_output)

        return text_features


class VisionLanguageModel(nn.Module):
    """
    Complete VLM architecture combining:
    - ViT for visual encoding (frozen)
    - GPT-2 for language generation (frozen)
    - Cross-attention layers for vision-language fusion (trainable)
    - Projection layers for dimension alignment (trainable)
    """
    def __init__(
        self,
        vit_model_name='google/vit-base-patch16-224',
        gpt2_model_name='gpt2',
        num_cross_attn_layers=6,  # Insert cross-attention every other GPT-2 layer
        freeze_vision=True,
        freeze_language=True
    ):
        super().__init__()

        #  Vision Encoder (ViT) 
        self.vision_encoder = ViTModel.from_pretrained(vit_model_name)
        self.vit_hidden_size = self.vision_encoder.config.hidden_size  # 768 for base

        if freeze_vision:
            # Freeze all ViT parameters - we only use it for feature extraction
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            print(f"✓ Vision encoder frozen ({vit_model_name})")

        #  Language Decoder (GPT-2) 
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        self.tokenizer.add_special_tokens({
            'pad_token': '<|PAD|>',
            'bos_token': '<|startoftext|>',  # Beginning of sequence
            # 'eos_token': '<EOS>'   # End of sequence
            # <|endoftext|> -> 505256
        })

        self.language_decoder = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.language_decoder.resize_token_embeddings(len(self.tokenizer))
        self.gpt2_hidden_size = self.language_decoder.config.n_embd  # 768 for base

        if freeze_language:
            # Freeze all GPT-2 parameters 
            for param in self.language_decoder.parameters():
                param.requires_grad = False
            print(f"✓ Language decoder frozen ({gpt2_model_name})")

        # UNFREEZE token and positional embeddings 
        self.language_decoder.transformer.wte.weight.requires_grad = True
        self.language_decoder.transformer.wpe.weight.requires_grad = True
        print("✓ Unfroze token and positional embeddings for special tokens")

        # Project ViT features to GPT-2 dimension if they don't match
        if self.vit_hidden_size != self.gpt2_hidden_size:
            self.vision_projection = nn.Linear(self.vit_hidden_size, self.gpt2_hidden_size)
            print(f"✓ Vision projection layer added: {self.vit_hidden_size} → {self.gpt2_hidden_size}")
        else:
            self.vision_projection = nn.Identity()

        # Create cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(
                hidden_dim=self.gpt2_hidden_size,
                num_heads=12,
                dropout=0.1
            )
            for _ in range(num_cross_attn_layers)
        ])
        print(f"✓ Added {num_cross_attn_layers} trainable cross-attention layers")

        total_gpt2_layers = self.language_decoder.config.n_layer  # 12 for GPT-2
        self.cross_attn_positions = [
            int(i * total_gpt2_layers / num_cross_attn_layers) + 1
            for i in range(num_cross_attn_layers)
        ]

        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\nParameters:")
        print(f"   Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"   Total: {total_params:,}\n")

    def forward(self, images, input_ids, attention_mask=None, labels=None):
        """
        Args:
            images: Batch of images (batch_size, 3, 224, 224)
            input_ids: Tokenized captions (batch_size, seq_len)
            attention_mask: Mask for padding tokens (batch_size, seq_len)
            labels: Ground truth tokens for loss calculation (batch_size, seq_len)

        Returns:
            Dictionary containing:
                - loss: Cross-entropy loss (if labels provided)
                - logits: Token prediction logits (batch_size, seq_len, vocab_size)
        """
        batch_size = images.shape[0]

        # Extract Visual Features 
        vision_outputs = self.vision_encoder(pixel_values=images) # (batch_size, num_patches + 1, vit_hidden_size), +1 for [CLS] token 
        image_features = vision_outputs.last_hidden_state

        # Project to GPT-2 dimension
        image_features = self.vision_projection(image_features) # (batch_size, num_patches + 1, gpt2_hidden_size)

        # Get Text Embeddings from GPT-2 
        text_embeds = self.language_decoder.transformer.wte(input_ids)  # Word embeddings
        position_embeds = self.language_decoder.transformer.wpe(
            torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        )
        hidden_states = text_embeds + position_embeds # (batch_size, seq_len, gpt2_hidden_size)
        # Process Through GPT-2 with Cross-Attention 
        cross_attn_idx = 0

        for i, block in enumerate(self.language_decoder.transformer.h):
            hidden_states = block(hidden_states)[0]
            # Inject cross-attention 
            if i in self.cross_attn_positions and cross_attn_idx < len(self.cross_attention_layers):
                hidden_states = self.cross_attention_layers[cross_attn_idx](
                    text_features=hidden_states,
                    image_features=image_features,
                    attention_mask=None  
                )
                cross_attn_idx += 1

        hidden_states = self.language_decoder.transformer.ln_f(hidden_states)
        logits = self.language_decoder.lm_head(hidden_states) # (batch_size, seq_len, vocab_size)

        # Calculate Loss 
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return {
            'loss': loss,
            'logits': logits
        }

    @torch.no_grad()
    def generate_caption(self, image, max_length=50, temperature=1.0, top_k=50):
        """
        Generate caption for a single image using greedy/sampling decoding.

        Args:
            image: Single image tensor (3, 224, 224)
            max_length: Maximum caption length
            temperature: Sampling temperature (1.0 = no change, <1 = more conservative)
            top_k: Consider only top-k tokens for sampling

        Returns:
            Generated caption string
        """
        self.eval()
        device = next(self.parameters()).device

        # Prepare image
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(device)

        # Extract image features
        vision_outputs = self.vision_encoder(pixel_values=image)
        image_features = vision_outputs.last_hidden_state
        image_features = self.vision_projection(image_features)

        # Start with BOS token
        generated = torch.tensor(
            [[self.tokenizer.bos_token_id]],
            device=device
        )

        # Generate tokens one by one
        for _ in range(max_length):
            # Get embeddings for current sequence
            text_embeds = self.language_decoder.transformer.wte(generated)
            position_embeds = self.language_decoder.transformer.wpe(
                torch.arange(generated.shape[1], device=device).unsqueeze(0)
            )
            hidden_states = text_embeds + position_embeds

            # Process through GPT-2 with cross-attention
            cross_attn_idx = 0
            for i, block in enumerate(self.language_decoder.transformer.h):
                hidden_states = block(hidden_states)[0]

                if i in self.cross_attn_positions and cross_attn_idx < len(self.cross_attention_layers):
                    hidden_states = self.cross_attention_layers[cross_attn_idx](
                        text_features=hidden_states,
                        image_features=image_features
                    )
                    cross_attn_idx += 1

            hidden_states = self.language_decoder.transformer.ln_f(hidden_states)
            logits = self.language_decoder.lm_head(hidden_states)

            # Get logits for next token (last position)
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) # greedy

            # Stop if EOS token generated
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

        caption = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return caption
