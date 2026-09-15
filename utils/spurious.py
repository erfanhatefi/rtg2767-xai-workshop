"""Utilities for the workshop's controlled ImageNet-W shortcut example."""

from pathlib import Path

import torch

from .vision import load_display_image


IMAGENET_W_CARTON_INDEX = 478
IMAGENET_W_EXAMPLE_WNID = "n03394916"
IMAGENET_W_EXAMPLE_FILENAME = "n03394916_68382.JPEG"


def resolve_imagenet_w_example_path(root, download=False):
    """Return the verified French-horn image used for the watermark demo.

    The image belongs to the Imagenette-320 validation split. Initializing the
    torchvision dataset provides checksum-verified download and extraction.
    """
    from torchvision.datasets import Imagenette

    dataset = Imagenette(
        root=root,
        split="val",
        size="320px",
        download=download,
    )
    matching_paths = [
        Path(image_path)
        for image_path, _ in dataset._samples
        if Path(image_path).name == IMAGENET_W_EXAMPLE_FILENAME
    ]
    if len(matching_paths) != 1:
        raise FileNotFoundError(
            "Could not uniquely resolve the workshop ImageNet-W example "
            f"{IMAGENET_W_EXAMPLE_FILENAME!r} below {Path(root).resolve()}"
        )
    return matching_paths[0]


def add_imagenet_w_watermark(image_tensor, selected_samples=None):
    """Apply the published ImageNet-W transform to selected display images.

    ``image_tensor`` may have shape ``[3, H, W]`` or ``[B, 3, H, W]`` and must
    be in display space ``[0, 1]``. ``selected_samples`` is an optional Boolean
    mask of length ``B``; omitted means that every sample receives a watermark.
    The result preserves the input device and dtype.
    """
    try:
        from imagenet_w import AddWatermark
    except ImportError as error:
        raise ImportError(
            "Install imagenet-w==1.0.0 to run the controlled shortcut example"
        ) from error

    image_tensor = torch.as_tensor(image_tensor)
    unbatched = image_tensor.ndim == 3
    image_batch = image_tensor.unsqueeze(0) if unbatched else image_tensor
    if image_batch.ndim != 4 or image_batch.shape[1] != 3:
        raise ValueError("image_tensor must have shape [3, H, W] or [B, 3, H, W]")
    if image_batch.shape[-2] != image_batch.shape[-1]:
        raise ValueError("ImageNet-W expects square images")

    num_samples = len(image_batch)
    if selected_samples is None:
        selected_samples = torch.ones(num_samples, dtype=torch.bool)
    selected_samples = torch.as_tensor(selected_samples, dtype=torch.bool).cpu()
    if selected_samples.shape != (num_samples,):
        raise ValueError("selected_samples must have shape [batch_size]")

    watermark_transform = AddWatermark(image_size=image_batch.shape[-1])
    output_images = []
    for sample_index, image in enumerate(image_batch):
        cpu_image = image.detach().float().cpu()
        if selected_samples[sample_index]:
            cpu_image = watermark_transform(cpu_image)
        output_images.append(cpu_image)

    output_batch = torch.stack(output_images).to(
        device=image_tensor.device,
        dtype=image_tensor.dtype,
    )
    return output_batch[0] if unbatched else output_batch


def load_spurious_display_image_batch(sample_records, device):
    """Load images and apply ImageNet-W according to each record's flag."""
    image_tensors = [
        load_display_image(record["path"])[0]
        for record in sample_records
    ]
    display_images = torch.stack(image_tensors)
    selected_samples = [
        record.get("imagenet_w_watermark", False)
        for record in sample_records
    ]
    display_images = add_imagenet_w_watermark(
        display_images,
        selected_samples=selected_samples,
    )
    return display_images.to(device)


def load_watermark_example(root, device, download=True):
    """Load the workshop's clean image and its ImageNet-W counterpart."""
    image_path = resolve_imagenet_w_example_path(root, download=download)
    clean_image = load_display_image(image_path).to(device)
    watermarked_image = add_imagenet_w_watermark(clean_image)
    return image_path, clean_image, watermarked_image


def load_watermarked_images(sample_records, device):
    """Load a cohort and apply each record's declared watermark intervention."""
    return load_spurious_display_image_batch(sample_records, device)
