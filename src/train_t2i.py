import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import List, Dict, Tuple
import os
from collections import defaultdict

# Import models
from src.models.multimodal_concept_mapper import MultimodalConceptMapper
from src.models.decoders import ConceptToImageVAE
from src.models.encoders import ImageEncoder, TextEncoder
from src.models.bag_of_concepts import BagOfConcepts

# Import losses
from src.losses import (
    t2i_vae_reconstruction_loss,
    t2i_vae_kl_loss,
)

# Import configs
from configs.bag_of_concepts_config import BagOfConceptsConfig
from configs.concept_mapper_config import ConceptMapperConfig
from configs.concept_to_image_vae_config import ConceptToImageVAEConfig
from configs.image_encoder_config import ImageEncoderConfig
from configs.text_encoder_config import TextEncoderConfig

# Import data loader
from data.loader import load_flickr_dataset

# --- Simple Caption Tokenizer (for demonstration) ---
# This is still needed for the MultimodalConceptMapper's text input, even if not for captioning output
class CaptionTokenizer:
    def __init__(self, special_tokens: List[str] = None):
        self.word_to_idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx_to_word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.vocab_size = len(self.word_to_idx)
        if special_tokens:
            for token in special_tokens:
                if token not in self.word_to_idx:
                    self.word_to_idx[token] = self.vocab_size
                    self.idx_to_word[self.vocab_size] = token
                    self.vocab_size += 1

    def build_vocabulary(self, captions: List[str]):
        for caption in captions:
            for word in caption.lower().split():
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = self.vocab_size
                    self.idx_to_word[self.vocab_size] = word
                    self.vocab_size += 1

    def tokenize(self, caption: str, max_len: int = None) -> List[int]:
        tokens = [self.word_to_idx.get(word, self.word_to_idx["<unk>"]) for word in caption.lower().split()]
        tokens = [self.word_to_idx["<sos>"]] + tokens + [self.word_to_idx["<eos>"]]
        
        if max_len and len(tokens) > max_len:
            tokens = tokens[:max_len-1] + [self.idx_to_word["<eos>"]]
        elif max_len and len(tokens) < max_len:
            tokens = tokens + [self.word_to_idx["<pad>"]] * (max_len - len(tokens))
        
        return tokens

    def detokenize(self, token_ids: List[int]) -> str:
        words = [self.idx_to_word.get(idx, "<unk>") for idx in token_ids]
        # Remove special tokens for display
        filtered_words = []
        for word in words:
            if word == "<eos>":
                break
            if word not in ["<pad>", "<sos>", "<unk>"]:
                filtered_words.append(word)
        return " ".join(filtered_words)

# --- Training Script for Text-to-Image Generation ---
def train_t2i(
    concept_mapper: MultimodalConceptMapper,
    image_vae_decoder: ConceptToImageVAE,
    dataloader: DataLoader,
    tokenizer: CaptionTokenizer, # Still needed for text processing in concept mapper
    optimizer_mapper: optim.Optimizer,
    optimizer_vae: optim.Optimizer,
    epochs: int,
    device: torch.device,
    caption_max_len: int, # Still needed for text processing in concept mapper
) -> Dict[str, List[float]]:
    concept_mapper.train()
    image_vae_decoder.train()

    all_vae_losses = []

    for epoch in range(epochs):
        total_vae_loss = 0

        for batch_idx, (images, captions) in enumerate(dataloader):
            images = images.to(device)
            
            # For text-to-image generation, we need a single caption per image
            # Flickr30k provides 5 captions per image, let's pick the first one for simplicity
            texts = [c[0] for c in captions] 

            # Generate dummy concept_ids for MultimodalConceptMapper
            concept_ids = torch.randint(0, BagOfConceptsConfig().num_concepts, (images.size(0),)).to(device)

            # --- Text-to-Image Generation Training ---
            optimizer_mapper.zero_grad()
            optimizer_vae.zero_grad()
            
            # Get concepts from the current batch (images and texts)
            # The MultimodalConceptMapper takes images, texts, concept_ids.
            # For text-to-image, we want concepts from text.
            # We'll pass the actual images and texts to the mapper.
            concepts_from_multimodal = concept_mapper(images, texts, concept_ids)

            # Generate images from these concepts
            generated_images = image_vae_decoder(concepts_from_multimodal)

            # Calculate VAE losses
            reconstruction_loss = t2i_vae_reconstruction_loss(generated_images, images)
            
            vae_loss = reconstruction_loss # + kl_loss (if available)
            
            vae_loss.backward()
            optimizer_mapper.step()
            optimizer_vae.step()
            total_vae_loss += vae_loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], "
                    f"VAE Loss: {vae_loss.item():.4f}"
                )
        
        avg_vae_loss = total_vae_loss / len(dataloader)
        all_vae_losses.append(avg_vae_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Average VAE Loss: {avg_vae_loss:.4f}")
        
    return {
        "vae_loss": all_vae_losses,
    }

# The main function is removed from here, as it will be called from run_experiment.py
# The CaptionTokenizer is kept here as it's a utility for this training script.
