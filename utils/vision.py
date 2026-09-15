"""Shared model, image, and preprocessing utilities for workshop notebooks."""

import urllib.request
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    VGG16_Weights,
    resnet18,
    vgg16,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FALLBACK_IMAGE_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
)


def get_imagenet_class_names():
    """Return the ordered class names used by torchvision ImageNet weights."""
    return list(ResNet18_Weights.DEFAULT.meta["categories"])


def resolve_imagenet_class_index(class_name):
    """Resolve a human-readable ImageNet class name to its model output index."""
    class_names = get_imagenet_class_names()
    try:
        return class_names.index(class_name)
    except ValueError as error:
        raise ValueError(
            f"Unknown ImageNet class {class_name!r}. Names are case-sensitive."
        ) from error


def download_file_if_missing(destination, source_url):
    """Download a fixed workshop asset only when it is not already available."""
    destination = Path(destination)
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        source_url,
        headers={"User-Agent": "xai-workshop-notebook"},
    )
    with urllib.request.urlopen(request) as response:
        destination.write_bytes(response.read())
    return destination


def resolve_example_image_path(
    local_image_path="dog_2.png",
    fallback_destination="workshop_assets/pytorch_dog.jpg",
):
    """Prefer a local workshop image and otherwise download the fallback image."""
    local_image_path = Path(local_image_path)
    if local_image_path.exists():
        return local_image_path
    return download_file_if_missing(fallback_destination, FALLBACK_IMAGE_URL)


def load_display_image(image_path, crop_size=224, resize_size=256):
    """Load one image as a batched RGB tensor in display space ``[0, 1]``."""
    display_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, antialias=True),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return display_transform(image).unsqueeze(0)


def normalize_imagenet(image_batch):
    """Normalize a ``[B, 3, H, W]`` display-space batch for ImageNet models."""
    channel_mean = torch.as_tensor(
        IMAGENET_MEAN,
        dtype=image_batch.dtype,
        device=image_batch.device,
    ).view(1, 3, 1, 1)
    channel_std = torch.as_tensor(
        IMAGENET_STD,
        dtype=image_batch.dtype,
        device=image_batch.device,
    ).view(1, 3, 1, 1)
    return (image_batch - channel_mean) / channel_std


def load_pretrained_resnet18(device):
    """Load evaluation-mode ImageNet ResNet-18 and return its class names."""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).eval().to(device)
    return model, get_imagenet_class_names()


def load_pretrained_vgg16(device):
    """Load evaluation-mode ImageNet VGG-16 and return its class names."""
    weights = VGG16_Weights.DEFAULT
    model = vgg16(weights=weights).eval().to(device)
    return model, list(weights.meta["categories"])


def predict_top_classes(model, image_batch, class_names, top_k=5):
    """Return top class indices, names, logits, and probabilities."""
    with torch.no_grad():
        logits = model(normalize_imagenet(image_batch))[0]
        probabilities = logits.softmax(dim=0)

    top_probabilities, top_indices = probabilities.topk(top_k)
    return [
        (
            int(index),
            class_names[int(index)],
            float(logits[index]),
            float(probability),
        )
        for probability, index in zip(top_probabilities, top_indices)
    ]


def predict_image_logits(image_arrays, model, model_device):
    """Map SHAP ``[B, H, W, 3]`` arrays in ``[0, 1]`` to class logits."""
    image_batch = torch.as_tensor(
        image_arrays,
        dtype=torch.float32,
        device=model_device,
    ).permute(0, 3, 1, 2)

    with torch.no_grad():
        logits = model(normalize_imagenet(image_batch))
    return logits.cpu().numpy()
