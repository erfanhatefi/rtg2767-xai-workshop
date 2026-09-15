"""Small perturbation methods used by the workshop notebooks."""

import torch

from .vision import normalize_imagenet


def _window_positions(length, patch_size, stride):
    """Return patch starts that also cover the final image pixels."""
    if not 0 < patch_size <= length:
        raise ValueError("patch_size must fit inside the image")
    if stride <= 0:
        raise ValueError("stride must be positive")

    positions = list(range(0, length - patch_size + 1, stride))
    final_position = length - patch_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def occlusion(
    model,
    image,
    target,
    baseline,
    patch_size=56,
    batch_size=64,
):
    """Return the target-logit drop caused by sliding-patch replacement.

    ``image`` and ``baseline`` have shape ``[1,3,H,W]`` in display space.
    The stride is half of ``patch_size``, giving 50% overlap between adjacent
    patches. The final patch is shifted when needed so the image border is
    always covered.

    Positive output values mean that replacing the region lowered the target
    score; negative values mean that replacement increased it.
    """
    if image.shape != baseline.shape or image.shape[0] != 1:
        raise ValueError("image and baseline must have matching [1,3,H,W] shapes")

    _, _, height, width = image.shape
    stride = max(1, patch_size // 2)
    regions = [
        (row, column)
        for row in _window_positions(height, patch_size, stride)
        for column in _window_positions(width, patch_size, stride)
    ]

    with torch.no_grad():
        original_score = model(normalize_imagenet(image))[0, target]

    score_sum = torch.zeros(
        (height, width),
        dtype=image.dtype,
        device=image.device,
    )
    overlap_count = torch.zeros_like(score_sum)

    # Evaluate many masked images together; this is the main speedup over a
    # patch-by-patch implementation.
    for start in range(0, len(regions), batch_size):
        batch_regions = regions[start : start + batch_size]
        masked_images = image.repeat(len(batch_regions), 1, 1, 1)
        for batch_index, (row, column) in enumerate(batch_regions):
            masked_images[
                batch_index,
                :,
                row : row + patch_size,
                column : column + patch_size,
            ] = baseline[
                0,
                :,
                row : row + patch_size,
                column : column + patch_size,
            ]

        with torch.no_grad():
            masked_scores = model(normalize_imagenet(masked_images))[:, target]

        for score, (row, column) in zip(original_score - masked_scores, batch_regions):
            score_sum[row : row + patch_size, column : column + patch_size] += score
            overlap_count[row : row + patch_size, column : column + patch_size] += 1

    heatmap = score_sum / overlap_count.clamp_min(1)
    return heatmap, float(original_score)
