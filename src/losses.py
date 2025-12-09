import torch
import torch.nn.functional as F

def vq_loss(concept: torch.Tensor, slots: torch.Tensor, commitment: float) -> torch.Tensor:
    codebook_loss = F.mse_loss(slots.detach(), concept)
    commitment_loss = F.mse_loss(slots, concept.detach())

    return codebook_loss + (commitment * commitment_loss)

def image_reconstruction_loss(image: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    # NOTE: Here we don't add the KL-Divergence loss because we have discrete concepts so we will use the vq_loss() function above
    return F.mse_loss(image, reconstructed, reduction="sum")


def text_reconstruction_loss(logits: torch.Tensor, targets: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    # NOTE: Here we permute as logits are (batch_size, seq_len, vocab_size)
    # but F.cross_entropy expect (batch_size, vocab_size, seq_len)
    logits = logits.permute(0, 2, 1)

    loss = F.cross_entropy(logits, targets, ignore_index=pad_token_id)

    return loss

