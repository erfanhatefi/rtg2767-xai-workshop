"""Readable gradient and CAM implementations for the vision workshop."""

from functools import partial

import torch

from .vision import normalize_imagenet


def input_gradient(model, image, target):
    """Differentiate one target score with respect to the input image."""
    differentiable_image = image.detach().clone().requires_grad_(True)
    target_score = model(normalize_imagenet(differentiable_image))[0, target]
    gradient = torch.autograd.grad(target_score, differentiable_image)[0]
    return gradient.detach(), float(target_score.detach())


def gradient_magnitude(gradient):
    """Reduce RGB gradients to one unsigned sensitivity map."""
    return gradient.abs().amax(dim=1)[0]


def gradient_times_input(image, gradient):
    """Reduce signed gradient × input contributions over RGB channels."""
    if image.shape != gradient.shape:
        raise ValueError("image and gradient must have identical shapes")
    return (image * gradient).sum(dim=1)[0]


def smoothgrad(model, image, target, samples=12, noise=0.10):
    """Average input gradients over Gaussian-noisy copies of one image."""
    noisy_images = image.expand(samples, -1, -1, -1)
    noisy_images = (noisy_images + noise * torch.randn_like(noisy_images)).clamp(0, 1)
    noisy_images = noisy_images.detach().requires_grad_(True)

    # Samples are independent, so one backward pass computes every gradient.
    target_score_sum = model(normalize_imagenet(noisy_images))[:, target].sum()
    gradients = torch.autograd.grad(target_score_sum, noisy_images)[0]
    return gradients.detach().mean(dim=0, keepdim=True)


def _save_activation(module, inputs, output, storage):
    """Store a layer output for CAM methods without changing the forward pass."""
    storage["activation"] = output


def _forward_with_activation(model, image, layer):
    """Run the model once and return its logits plus one layer activation."""
    storage = {}
    hook = layer.register_forward_hook(partial(_save_activation, storage=storage))
    try:
        logits = model(normalize_imagenet(image))
    finally:
        hook.remove()
    return logits, storage["activation"]


def cam(model, image, target, layer):
    """Compute signed CAM for a ResNet global-average-pooling head."""
    logits, activation = _forward_with_activation(model, image, layer)
    class_weights = model.fc.weight[target]
    heatmap = torch.einsum("c,chw->hw", class_weights, activation[0])

    bias = model.fc.bias[target] if model.fc.bias is not None else 0.0
    reconstructed_score = heatmap.mean() + bias
    return (
        heatmap.detach(),
        float(logits[0, target].detach()),
        float(reconstructed_score.detach()),
    )


def gradcam(model, image, target, layer):
    """Compute positive Grad-CAM for one target score and feature layer."""
    logits, activation = _forward_with_activation(model, image, layer)
    target_score = logits[0, target]
    activation_gradient = torch.autograd.grad(target_score, activation)[0]

    # Spatially averaged gradients become one importance weight per channel.
    channel_weights = activation_gradient.mean(dim=(2, 3), keepdim=True)
    heatmap = (channel_weights * activation).sum(dim=1).relu()[0]
    return heatmap.detach(), float(target_score.detach())
