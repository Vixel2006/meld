import torch

def train(model, optimizer, dataloader, config):
    for epoch in range(config.epochs):
        for imgs, txts in dataloader:
            pass

