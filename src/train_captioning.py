import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import List, Dict, Tuple
import os
from collections import defaultdict
from dataclasses import dataclass

# Import models
from src.models.multimodal_concept_mapper import MultimodalConceptMapper
from src.models.decoders import ConceptConditionedImageCaptioning
from src.models.encoders import ImageEncoder, TextEncoder
from src.models.bag_of_concepts import BagOfConcepts
from src.models.transformer_blocks import PositionalEncoding, TransformerDecoder # Import original TransformerDecoder

# Import losses
from src.losses import captioning_cross_entropy_loss

# Import configs
from configs.bag_of_concepts_config import BagOfConceptsConfig
from configs.captioning_config import CaptioningConfig
from configs.concept_mapper_config import ConceptMapperConfig
from configs.image_encoder_config import ImageEncoderConfig
from configs.text_encoder_config import TextEncoderConfig

# Import data loader
from data.loader import load_flickr_dataset

# --- Simple Caption Tokenizer ---
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
        words = [self.idx_to_idx.get(idx, "<unk>") for idx in token_ids]
        # Remove special tokens for display
        filtered_words = []
        for word in words:
            if word == "<eos>":
                break
            if word not in ["<pad>", "<sos>", "<unk>"]:
                filtered_words.append(word)
        return " ".join(filtered_words)

# --- Training Script for Image Captioning ---
def train_captioning(
    concept_mapper: MultimodalConceptMapper,
    captioning_decoder: ConceptConditionedImageCaptioning,
    dataloader: DataLoader,
    tokenizer: CaptionTokenizer,
    optimizer_mapper: optim.Optimizer,
    optimizer_captioning: optim.Optimizer,
    epochs: int,
    device: torch.device,
    caption_max_len: int,
) -> Dict[str, List[float]]:
    concept_mapper.train()
    captioning_decoder.train()

    captioning_losses = []

    print(f"Starting Image Captioning Training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, captions_list) in enumerate(dataloader):
            images = images.to(device)

            # Flatten the list of lists of captions and tokenize
            flat_captions = [caption for sublist in captions_list for caption in sublist]
            
            # Tokenize all captions and pad them
            tokenized_captions = [tokenizer.tokenize(cap, caption_max_len) for cap in flat_captions]
            captions_input_ids = torch.tensor(tokenized_captions, dtype=torch.long, device=device)

            # Separate input and target for teacher forcing
            # Input will be <sos>, w1, w2, ..., wn
            # Target will be w1, w2, ..., wn, <eos>
            decoder_input_ids = captions_input_ids[:, :-1]
            target_caption_ids = captions_input_ids[:, 1:]

            # Create caption lengths (excluding <eos> for input, and <sos> for target)
            # Find the first <pad> token or use max_len if no pad token
            captions_lengths = torch.sum(target_caption_ids != tokenizer.word_to_idx["<pad>"], dim=1)

            optimizer_mapper.zero_grad()
            optimizer_captioning.zero_grad()

            # Get concept vectors from images using the concept mapper
            # Dummy concept_ids for the mapper, as it's not used in this path
            dummy_concept_ids = torch.randint(0, concept_mapper.bag_of_concepts.num_concepts, (images.size(0),)).to(device)
            concept_vectors = concept_mapper(images, None, dummy_concept_ids) # (batch_size, concept_dim)

            # Forward pass through the captioning decoder
            logits = captioning_decoder(concept_vectors, decoder_input_ids, captions_lengths)

            # Reshape logits and targets for loss calculation
            # logits: (batch_size * seq_len, vocab_size)
            # targets: (batch_size * seq_len)
            loss = captioning_cross_entropy_loss(
                logits.reshape(-1, logits.size(-1)),
                target_caption_ids.reshape(-1),
                ignore_index=tokenizer.word_to_idx["<pad>"]
            )

            loss.backward()
            optimizer_mapper.step()
            optimizer_captioning.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        captioning_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Captioning Loss: {avg_loss:.4f}")

    return {
        "captioning_loss": captioning_losses,
    }
