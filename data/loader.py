import os
import torch
from torchvision.datasets import Flickr30k
from torch.utils.data import DataLoader
from torchvision import transforms


def load_flickr_dataset(
    root_dir: str = "./data",
    ann_file: str = "flickr30k_annotations.json",
    split: str = "train",
    transform=None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Construct full path to annotations file
    full_ann_path = os.path.join(root_dir, ann_file)

    # The Flickr30K dataset in torchvision expects the images in a subdirectory
    # named 'flickr30k_images' within the root_dir, and the annotations file
    # to be directly in the root_dir or specified.
    # For simplicity, let's assume images are in root_dir/flickr30k_images
    # and the annotation file is in root_dir.
    # You might need to adjust `root` and `ann_file` based on how you download/structure the dataset.
    dataset = Flickr30k(
        root=os.path.join(root_dir, "flickr30k_images"), # Assuming images are in this subfolder
        ann_file=full_ann_path,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader

