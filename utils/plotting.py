"""Blue-white-red plotting helpers for vision attribution workshops."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as torch_functional

from lexplainet.heatmap import draw_heatmap_on_axis, prepare_heatmap


def _display_image_array(image_batch):
    """Convert a display-space image tensor to a Matplotlib RGB array."""
    if image_batch.ndim == 4:
        image_batch = image_batch[0]
    return image_batch.detach().cpu().permute(1, 2, 0).numpy()


def _resize_heatmap(heatmap, output_shape):
    """Resize a normalized 2D heatmap to ``output_shape`` when necessary."""
    if tuple(heatmap.shape) == tuple(output_shape):
        return heatmap

    heatmap_tensor = torch.as_tensor(heatmap, dtype=torch.float32)[None, None]
    resized_heatmap = torch_functional.interpolate(
        heatmap_tensor,
        size=output_shape,
        mode="bilinear",
        align_corners=False,
    )
    return resized_heatmap[0, 0].numpy()


def show_image(image_batch, title=None, figure_size=(6, 6)):
    """Display one ``[1, 3, H, W]`` image tensor."""
    figure, axis = plt.subplots(figsize=figure_size)
    axis.imshow(_display_image_array(image_batch))
    if title is not None:
        axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    plt.show()
    return figure, axis


def show_image_grid(image_batches, titles):
    """Display equally sized image tensors in one row."""
    if len(image_batches) != len(titles):
        raise ValueError("image_batches and titles must have the same length")

    figure, axes = plt.subplots(
        1,
        len(image_batches),
        figsize=(5 * len(image_batches), 5),
    )
    axes = np.atleast_1d(axes)
    for axis, image_batch, title in zip(axes, image_batches, titles):
        axis.imshow(_display_image_array(image_batch))
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    plt.show()
    return figure, axes


def plot_bwr_attributions(
    image_batch,
    attributions,
    titles,
    q=99,
    edge_overlay=True,
    shared_scale=False,
):
    """Plot attribution maps using LeXplaiNet's centered ``bwr`` convention.

    Blue denotes negative values, white zero, and red positive values. Maps are
    clipped symmetrically at the requested absolute percentile before plotting.
    ``shared_scale=True`` uses the largest panel scale for every map, allowing
    attribution magnitudes to be compared across panels.
    """
    if len(attributions) != len(titles):
        raise ValueError("attributions and titles must have the same length")

    image_array = _display_image_array(image_batch)
    image_shape = image_array.shape[:2]
    figure, axes = plt.subplots(
        1,
        len(attributions) + 1,
        figsize=(5 * (len(attributions) + 1), 5),
    )
    axes = np.atleast_1d(axes)

    axes[0].imshow(image_array)
    axes[0].set_title("Input")
    axes[0].axis("off")

    prepared_heatmaps = []
    raw_scales = []
    for attribution in attributions:
        prepared_heatmap, raw_scale = prepare_heatmap(attribution, q=q)
        prepared_heatmap = _resize_heatmap(prepared_heatmap, image_shape)
        prepared_heatmaps.append(prepared_heatmap)
        raw_scales.append(raw_scale)

    if shared_scale:
        largest_scale = max(raw_scales)
        prepared_heatmaps = [
            heatmap * raw_scale / largest_scale
            for heatmap, raw_scale in zip(prepared_heatmaps, raw_scales)
        ]

    for axis, prepared_heatmap, title in zip(
        axes[1:],
        prepared_heatmaps,
        titles,
    ):
        draw_heatmap_on_axis(
            axis,
            prepared_heatmap,
            cmap="bwr",
            image=image_batch,
            edge_overlay=edge_overlay,
            edge_threshold=0.9,
            edge_alpha=0.2,
            edge_color="gray",
        )
        axis.set_title(title)

    figure.tight_layout()
    plt.show()
    return figure, axes, raw_scales
