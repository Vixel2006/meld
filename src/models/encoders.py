import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPVisionModel, RobertaModel

class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model = CLIPVisionModel.from_pretrained(model_name)

        # Freezing the pre-trained image encoder
        for p in self.model.parameters(): p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # NOTE: Here we just get the final hidden layer of the encoding,
        # we should study the possibility of pooling different layers results for more compact representations
        output = self.model(pixel_values=images, return_tensors="pt").last_hidden_state
        return output


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "distilroberta-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)

        # Freezing the pre-trained text encoder
        for p in self.model.parameters(): p.requires_grad = False

    def forward(self, texts: torch.Tensor) -> torch.Tensor:
        input_ids, attn_masks = **self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


        # NOTE: Here we just get the final hidden layer of the encoding,
        # we should study the possibility of pooling different layers results for more compact representations
        output = self.model(input_ids=input_ids, attention_masks=attn_masks).last_hidden_state
        return output
