import torch
import torch.nn.functional as F

def cosine_similarity(tensor1, tensor2):
    return (tensor1 * tensor2).sum(dim=-1)

def captioning_cross_entropy_loss(predictions, targets, ignore_index=-100):
    # Reshape for F.cross_entropy: (N, C) for predictions, (N) for targets
    predictions = predictions.view(-1, predictions.size(-1))
    targets = targets.view(-1)
    return F.cross_entropy(predictions, targets, ignore_index=ignore_index)

def captioning_clip_loss(image_features, text_features):
    # Assuming features are already L2-normalized by the CLIP encoder
    similarity = cosine_similarity(image_features, text_features)
    # Minimize negative similarity to maximize similarity
    return -similarity.mean()

def t2i_perceptual_loss(generated_images, target_images, feature_extractor):
    gen_features = feature_extractor(generated_images)
    target_features = feature_extractor(target_images)
    # Assuming feature_extractor returns a list of feature maps from different layers
    # We sum the L1/L2 loss across all feature maps
    loss = 0
    for gen_f, target_f in zip(gen_features, target_features):
        loss += F.l1_loss(gen_f, target_f) # L1 is common for perceptual loss
    return loss

def t2i_vae_reconstruction_loss(reconstructed_images, target_images):
    # Assuming images are normalized to [0, 1] or [-1, 1]
    # For pixel values in [0, 1], BCEWithLogitsLoss or MSE is common.
    # For pixel values in [-1, 1], MSE is common.
    return F.mse_loss(reconstructed_images, target_images)

def t2i_vae_kl_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0) # Mean over batch

def t2i_clip_loss(text_features, image_features):
    # Assuming features are already L2-normalized by the CLIP encoder
    similarity = cosine_similarity(text_features, image_features)
    # Minimize negative similarity to maximize similarity
    return -similarity.mean()
