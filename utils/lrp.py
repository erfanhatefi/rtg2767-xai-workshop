"""Compact LRP and CRP helpers for the workshop notebooks."""

import contextlib
import io

import torch

from lexplainet.implicit.composite_core import patch_composite, undo_patch_all
from lexplainet.implicit.model_specific_patches import (
    canonize_resnet,
)
from lexplainet.implicit.rules import (
    epsilon_rule_non_zero,
    identity_rule,
    zplus_rule,
)

from .vision import normalize_imagenet


IMAGE_LRP_RULES = {
    torch.nn.Conv2d: (zplus_rule, {"ignore_bias": True}),
    torch.nn.Linear: (
        epsilon_rule_non_zero,
        {"epsilon": 1e-6, "ignore_bias": True},
    ),
    torch.nn.ReLU: (identity_rule, {}),
}


def setup_lrp(model, resnet=False):
    """Prepare a CNN for the workshop's implicit-LRP rule configuration."""
    with contextlib.redirect_stdout(io.StringIO()):
        undo_patch_all(model)
        if resnet:
            # Folding BatchNorm into adjacent convolutions gives LRP explicit
            # affine layers to redistribute through.
            canonize_resnet(model)
        patch_composite(model, IMAGE_LRP_RULES)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class _LayerCapture:
    """Capture one layer activation and optionally keep one CRP channel.

    PyTorch calls this object during the forward pass. Keeping ``channel`` and
    ``activation`` together avoids partially applied callback functions and
    makes the hook's state explicit.
    """

    def __init__(self, channel=None):
        self.channel = channel
        self.activation = None

    def keep_selected_channel(self, gradient):
        """Zero the backward signal outside the selected channel."""
        masked_gradient = torch.zeros_like(gradient)
        masked_gradient[:, self.channel] = gradient[:, self.channel]
        return masked_gradient

    def __call__(self, _module, _inputs, output):
        """Store the layer output when PyTorch executes the forward hook."""
        if output.ndim != 4:
            raise ValueError(
                "CRP expects a [batch, channels, height, width] output"
            )
        if self.channel is not None and not 0 <= self.channel < output.shape[1]:
            raise ValueError(f"Invalid channel {self.channel}")

        self.activation = output
        output.retain_grad()
        if self.channel is not None:
            output.register_hook(self.keep_selected_channel)


def explain_image(model, image, target, layer=None, channel=None):
    """Return signed input relevance, optionally through one CRP channel."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("explain_image expects one [1,3,H,W] image")
    if channel is not None and layer is None:
        raise ValueError("channel requires a layer")

    model.zero_grad(set_to_none=True)
    model_input = normalize_imagenet(image).detach().clone().requires_grad_(True)

    layer_capture = None
    hook = None
    if layer is not None:
        layer_capture = _LayerCapture(channel=channel)
        hook = layer.register_forward_hook(layer_capture)
    try:
        target_score = model(model_input)[0, target]
    finally:
        if hook is not None:
            hook.remove()

    target_score.backward()
    input_relevance = model_input.detach() * model_input.grad.detach()
    result = {
        "heatmap": input_relevance.sum(dim=1)[0].float().cpu(),
        "input_sum": float(input_relevance.sum()),
        "target_score": float(target_score.detach()),
    }

    if layer is not None:
        activation = layer_capture.activation
        layer_relevance = activation.detach() * activation.grad.detach()
        result["channel_scores"] = (
            layer_relevance.sum(dim=(0, 2, 3)).float().cpu()
        )
    return result


def relevance_gap(result):
    """Return the relative gap between input relevance and target score."""
    target_score = result["target_score"]
    return abs(result["input_sum"] - target_score) / (abs(target_score) + 1e-12)
