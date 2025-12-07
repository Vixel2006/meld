import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from transformers import CLIPProcessor, CLIPModel
from torchvision import models, transforms
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Import project encoders for CLIP score calculation
from src.models.encoders import ImageEncoder, TextEncoder
from configs.image_encoder_config import ImageEncoderConfig
from configs.text_encoder_config import TextEncoderConfig

def calculate_clip_score(images: torch.Tensor, texts: List[str], device: torch.device) -> torch.Tensor:
    # Initialize ImageEncoder and TextEncoder (which use CLIP internally)
    # These encoders are designed to output target_hidden_size, which should be compatible.
    image_encoder = ImageEncoder(config=ImageEncoderConfig()).to(device)
    text_encoder = TextEncoder(config=TextEncoderConfig()).to(device)

    # Ensure models are in evaluation mode
    image_encoder.eval()
    text_encoder.eval()

    with torch.no_grad():
        # Get image features
        # ImageEncoder's forward returns (batch, seq_len_img, target_hidden_size)
        # For CLIP score, we typically want the pooled output, which is usually the [CLS] token or mean pooling.
        # The current ImageEncoder returns `last_hidden_state`. We need to decide how to pool it.
        # For CLIP, the vision transformer typically has a [CLS] token at the first position.
        # Assuming the ImageEncoder's output for CLIP is already pooled or the first token is the pooled representation.
        image_features = image_encoder(images).squeeze(1) # Assuming it returns (batch, 1, hidden_size)

        # Get text features
        # TextEncoder's forward returns (batch, seq_len_txt, target_hidden_size)
        # For text, we typically use the [CLS] token or mean pooling.
        # The TextEncoder uses RobertaModel, which also has a [CLS] token at the first position.
        # Assuming the TextEncoder's output for CLIP is already pooled or the first token is the pooled representation.
        # The TextEncoder's forward method takes `texts` (list of strings) directly.
        # It returns `last_hidden_state`. We need to pool it.
        # For RoBERTa, the first token is usually the CLS token.
        encoded_input = text_encoder.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        text_output = text_encoder.model(**encoded_input).last_hidden_state
        text_features = text_output[:, 0, :] # Get CLS token embedding
        if hasattr(text_encoder, 'proj'):
            text_features = text_encoder.proj(text_features)


        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Calculate cosine similarity
        clip_scores = (image_features * text_features).sum(dim=-1)
    
    return clip_scores

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, requires_grad=False, device='cpu'):
        super(VGGPerceptualLoss, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.to(device)

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, X):
        # Normalize input images to VGG's expected range
        X = self.normalize(X)
        h_relu1_2 = self.slice1(X)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_3 = self.slice3(h_relu2_2)
        h_relu4_3 = self.slice4(h_relu3_3)
        h_relu5_3 = self.slice5(h_relu4_3)
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3

def calculate_image_perceptual_similarity(
    images1: torch.Tensor, images2: torch.Tensor, device: torch.device
) -> torch.Tensor:
    perceptual_loss_model = VGGPerceptualLoss(device=device).to(device)
    perceptual_loss_model.eval()

    with torch.no_grad():
        features1 = perceptual_loss_model(images1)
        features2 = perceptual_loss_model(images2)

        loss = 0
        for f1, f2 in zip(features1, features2):
            loss += F.l1_loss(f1, f2, reduction='mean') # Using L1 loss on features

    return loss

def calculate_caption_metrics(
    generated_captions: List[str], reference_captions: List[List[str]]
) -> Dict[str, float]:
    if len(generated_captions) != len(reference_captions):
        raise ValueError("Number of generated captions must match number of reference caption sets.")

    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []

    chencherry = SmoothingFunction() # For handling short sentences

    for gen_cap, ref_caps in zip(generated_captions, reference_captions):
        tokenized_gen_cap = word_tokenize(gen_cap.lower())
        tokenized_ref_caps = [word_tokenize(ref.lower()) for ref in ref_caps]

        bleu1_scores.append(sentence_bleu(tokenized_ref_caps, tokenized_gen_cap, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1))
        bleu2_scores.append(sentence_bleu(tokenized_ref_caps, tokenized_gen_cap, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1))
        bleu3_scores.append(sentence_bleu(tokenized_ref_caps, tokenized_gen_cap, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1))
        bleu4_scores.append(sentence_bleu(tokenized_ref_caps, tokenized_gen_cap, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1))

    return {
        "BLEU-1": np.mean(bleu1_scores),
        "BLEU-2": np.mean(bleu2_scores),
        "BLEU-3": np.mean(bleu3_scores),
        "BLEU-4": np.mean(bleu4_scores),
    }

def calculate_concept_sparsity(concept_vectors: torch.Tensor) -> Dict[str, float]:
    if concept_vectors.numel() == 0:
        return {"sparsity_ratio": 0.0}

    # Example: Percentage of zero elements
    zero_elements = (concept_vectors == 0).sum().item()
    total_elements = concept_vectors.numel()
    sparsity_ratio = zero_elements / total_elements

    # Example: Average L1 norm (can indicate how "dense" the vectors are)
    avg_l1_norm = torch.mean(torch.norm(concept_vectors, p=1, dim=-1)).item()

    return {
        "sparsity_ratio": sparsity_ratio,
        "average_l1_norm": avg_l1_norm,
    }

def calculate_cross_modal_similarity_matrix(
    image_features: torch.Tensor, text_features: torch.Tensor
) -> torch.Tensor:
    # Normalize features
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    # Calculate cosine similarity matrix
    similarity_matrix = torch.matmul(image_features, text_features.T)
    return similarity_matrix
