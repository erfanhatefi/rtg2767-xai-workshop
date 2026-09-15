"""Dataset, cache, and interactive-browser utilities for the SpRAy workshop."""

import base64
import contextlib
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .spurious import add_imagenet_w_watermark, load_spurious_display_image_batch
from .vision import (
    load_display_image,
    load_pretrained_resnet18,
    normalize_imagenet,
    resolve_imagenet_class_index,
)


IMAGENETTE_WNID_TO_IMAGENET_INDEX = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}


def select_imagenette_class_samples(
    root,
    target_wnid,
    max_samples,
    split="val",
    size="320px",
    download=False,
    seed=7,
):
    """Select a deterministic sample from one Imagenette class.

    Returns a list of image records and metadata containing the matching
    1,000-class ImageNet target index.
    """
    from torchvision.datasets import Imagenette

    if target_wnid not in IMAGENETTE_WNID_TO_IMAGENET_INDEX:
        valid_wnids = ", ".join(IMAGENETTE_WNID_TO_IMAGENET_INDEX)
        raise ValueError(f"Unknown Imagenette WNID. Choose one of: {valid_wnids}")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    dataset = Imagenette(
        root=root,
        split=split,
        size=size,
        download=download,
    )
    dataset_class_index = dataset.wnid_to_idx[target_wnid]
    matching_indices = [
        index
        for index, (_, label) in enumerate(dataset._samples)
        if label == dataset_class_index
    ]

    random_generator = random.Random(seed)
    random_generator.shuffle(matching_indices)
    selected_indices = matching_indices[: min(max_samples, len(matching_indices))]
    records = [
        {
            "dataset_index": index,
            "path": dataset._samples[index][0],
            "wnid": target_wnid,
        }
        for index in selected_indices
    ]

    class_names = dataset.classes[dataset_class_index]
    metadata = {
        "dataset": "Imagenette",
        "dataset_size": size,
        "split": split,
        "target_wnid": target_wnid,
        "target_name": class_names[0],
        "target_aliases": list(class_names),
        "imagenet_target_index": IMAGENETTE_WNID_TO_IMAGENET_INDEX[target_wnid],
        "available_class_samples": len(matching_indices),
        "selected_samples": len(selected_indices),
        "seed": seed,
    }
    return records, metadata


def load_display_image_batch(image_paths, device):
    """Load center-cropped images as one display-space batch in ``[0, 1]``."""
    image_tensors = [load_display_image(path)[0] for path in image_paths]
    return torch.stack(image_tensors).to(device)


def save_spray_cache(cache_path, heatmaps, sample_records, metadata):
    """Save full-resolution heatmaps and JSON-compatible analysis metadata."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_array = torch.as_tensor(heatmaps).detach().float().cpu().numpy()
    np.savez_compressed(
        cache_path,
        heatmaps=heatmap_array,
        sample_records_json=json.dumps(sample_records),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    return cache_path


def load_spray_cache(cache_path):
    """Load heatmaps, sample records, and metadata without pickle objects."""
    with np.load(cache_path, allow_pickle=False) as archive:
        heatmaps = torch.from_numpy(archive["heatmaps"].copy())
        sample_records = json.loads(archive["sample_records_json"].item())
        metadata = json.loads(archive["metadata_json"].item())
    return heatmaps, sample_records, metadata


def make_spray_cohort(
    root,
    class_wnid,
    max_samples=120,
    watermark_fraction=0.25,
    download=False,
    seed=7,
):
    """Select one Imagenette class and watermark a seeded sample fraction."""
    if not 0.0 <= watermark_fraction <= 1.0:
        raise ValueError("watermark_fraction must be between 0 and 1")

    records, metadata = select_imagenette_class_samples(
        root=root,
        target_wnid=class_wnid,
        max_samples=max_samples,
        split="val",
        size="320px",
        download=download,
        seed=seed,
    )
    num_watermarked = round(watermark_fraction * len(records))
    generator = np.random.default_rng(seed)
    selected_indices = set(
        generator.choice(
            len(records),
            size=num_watermarked,
            replace=False,
        ).tolist()
    )

    cohort = []
    for sample_index, record in enumerate(records):
        enriched_record = dict(record)
        enriched_record["imagenet_w_watermark"] = sample_index in selected_indices
        cohort.append(enriched_record)

    metadata = dict(metadata)
    metadata["watermark_fraction"] = watermark_fraction
    metadata["watermarked_samples"] = num_watermarked
    return cohort, metadata


def get_spray_heatmaps(
    sample_records,
    dataset_class_wnid,
    target_name,
    cache_root,
    device,
    batch_size=16,
    reuse_cache=True,
    seed=7,
):
    """Load cached fixed-target LRP maps or compute and cache them once."""
    target_index = resolve_imagenet_class_index(target_name)
    target_slug = target_name.replace(" ", "_")
    num_watermarked = sum(
        record["imagenet_w_watermark"] for record in sample_records
    )
    cache_path = Path(cache_root) / (
        f"spray_resnet18_{dataset_class_wnid}_{target_slug}_"
        f"wm{num_watermarked}_{len(sample_records)}_seed{seed}.npz"
    )
    metadata = {
        "model": "torchvision-resnet18-default",
        "preprocessing": "resize-256_center-crop-224_imagenet-normalization",
        "lrp_composite": "conv-zplus_linear-epsilon-1e-6_relu-identity_no-bias",
        "dataset_class_wnid": dataset_class_wnid,
        "explanation_target_name": target_name,
        "explanation_target_index": target_index,
        "dataset_indices": [record["dataset_index"] for record in sample_records],
        "imagenet_w_watermark": [
            record["imagenet_w_watermark"] for record in sample_records
        ],
    }

    if reuse_cache and cache_path.exists():
        heatmaps, cached_records, cached_metadata = load_spray_cache(cache_path)
        current_paths_exist = all(
            Path(record["path"]).exists() for record in sample_records
        )
        same_number_of_records = len(cached_records) == len(sample_records)
        cache_matches_experiment = (
            cached_metadata == metadata
            and current_paths_exist
            and same_number_of_records
        )
        if cache_matches_experiment:
            # Cached paths refer to the machine that created the archive. The
            # metadata equality above verifies the ordered dataset indices and
            # watermark assignments, so only the machine-specific path changes.
            portable_records = []
            for cached_record, current_record in zip(cached_records, sample_records):
                portable_record = dict(cached_record)
                portable_record["path"] = current_record["path"]
                portable_records.append(portable_record)

            return heatmaps, portable_records, {
                "cache_path": cache_path,
                "loaded_from_cache": True,
                "target_index": target_index,
                "seconds": 0.0,
            }

    start = time.perf_counter()
    model, class_names = load_pretrained_resnet18(device)
    configure_resnet_for_spray_lrp(model)
    heatmaps, enriched_records = extract_dataset_lrp(
        model=model,
        sample_records=sample_records,
        target_index=target_index,
        target_name=target_name,
        imagenet_class_names=class_names,
        device=device,
        batch_size=batch_size,
    )
    save_spray_cache(cache_path, heatmaps, enriched_records, metadata)
    return heatmaps, enriched_records, {
        "cache_path": cache_path,
        "loaded_from_cache": False,
        "target_index": target_index,
        "seconds": time.perf_counter() - start,
    }


def configure_resnet_for_spray_lrp(model):
    """Configure a torchvision ResNet with the workshop's implicit-LRP rules."""
    from lexplainet.implicit.composite_core import patch_composite, undo_patch_all
    from lexplainet.implicit.model_specific_patches import canonize_resnet
    from lexplainet.implicit.rules import (
        epsilon_rule_non_zero,
        identity_rule,
        zplus_rule,
    )

    composite = {
        torch.nn.Conv2d: (zplus_rule, {"ignore_bias": True}),
        torch.nn.Linear: (
            epsilon_rule_non_zero,
            {"epsilon": 1e-6, "ignore_bias": True},
        ),
        torch.nn.ReLU: (identity_rule, {}),
    }
    with contextlib.redirect_stdout(io.StringIO()):
        undo_patch_all(model)
        canonize_resnet(model)
        patch_composite(model, composite)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def compute_lrp_heatmap_batch(model, display_images, target_index):
    """Compute signed input relevance for one fixed output and image batch."""
    model.zero_grad(set_to_none=True)
    model_inputs = normalize_imagenet(display_images).detach().clone()
    model_inputs.requires_grad_(True)

    logits = model(model_inputs)
    # Summing independent sample logits permits one batched backward call.
    target_logit_sum = logits[:, target_index].sum()
    modified_gradients = torch.autograd.grad(target_logit_sum, model_inputs)[0]
    input_relevance = model_inputs.detach() * modified_gradients.detach()
    signed_heatmaps = input_relevance.sum(dim=1).float().cpu()

    probabilities = logits.detach().softmax(dim=1)
    predicted_probabilities, predicted_indices = probabilities.max(dim=1)
    return {
        "heatmaps": signed_heatmaps,
        "target_logits": logits[:, target_index].detach().float().cpu(),
        "predicted_indices": predicted_indices.cpu(),
        "predicted_probabilities": predicted_probabilities.float().cpu(),
    }


def extract_dataset_lrp(
    model,
    sample_records,
    target_index,
    target_name,
    imagenet_class_names,
    device,
    batch_size,
):
    """Extract fixed-target LRP maps and prediction metadata in batches."""
    all_heatmaps = []
    enriched_records = []
    num_samples = len(sample_records)

    for batch_start in range(0, num_samples, batch_size):
        batch_records = sample_records[batch_start : batch_start + batch_size]
        display_images = load_spurious_display_image_batch(batch_records, device)
        batch_result = compute_lrp_heatmap_batch(
            model,
            display_images,
            target_index,
        )
        all_heatmaps.append(batch_result["heatmaps"])

        for index_in_batch, record in enumerate(batch_records):
            predicted_index = int(batch_result["predicted_indices"][index_in_batch])
            enriched_record = dict(record)
            enriched_record.update(
                {
                    "target_name": target_name,
                    "target_index": target_index,
                    "target_logit": float(
                        batch_result["target_logits"][index_in_batch]
                    ),
                    "predicted_index": predicted_index,
                    "predicted_name": imagenet_class_names[predicted_index],
                    "predicted_probability": float(
                        batch_result["predicted_probabilities"][index_in_batch]
                    ),
                }
            )
            enriched_records.append(enriched_record)

        num_finished = min(batch_start + batch_size, num_samples)
        print(f"Explained {num_finished:3d}/{num_samples} images", end="\r")

    print()
    return torch.cat(all_heatmaps, dim=0), enriched_records


def prepare_spray_features(heatmaps, output_size=20, normalize_each=True):
    """Pool ``[N,H,W]`` maps and return ``[N,output_size**2]`` vectors."""
    import torch.nn.functional as torch_functional

    heatmaps = torch.as_tensor(heatmaps, dtype=torch.float32)
    if heatmaps.ndim != 3:
        raise ValueError("heatmaps must have shape [samples, height, width]")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if not torch.isfinite(heatmaps).all():
        raise ValueError("heatmaps contains non-finite values")

    # SpRAy seeks broad, recurring relevance patterns. Area pooling suppresses
    # single-pixel variation and makes pairwise distances much cheaper.
    pooled_heatmaps = torch_functional.adaptive_avg_pool2d(
        heatmaps[:, None],
        output_size=(output_size, output_size),
    )[:, 0]
    feature_vectors = pooled_heatmaps.flatten(start_dim=1).cpu().numpy()
    if normalize_each:
        # With row normalization, distance describes spatial pattern rather
        # than allowing total relevance magnitude to dominate the graph.
        vector_norms = np.linalg.norm(feature_vectors, axis=1, keepdims=True)
        feature_vectors = feature_vectors / np.maximum(vector_norms, 1e-12)
    return feature_vectors, pooled_heatmaps


def run_spray_spectral_clustering(
    feature_vectors,
    num_neighbors=10,
    max_clusters=8,
    num_clusters=None,
    seed=7,
):
    """Cluster explanation vectors through a normalized k-NN graph.

    The returned dictionary exposes the eigenvalues and eigengaps needed for
    interpretation while keeping graph construction out of the notebook.
    """
    from scipy.sparse.csgraph import laplacian as graph_laplacian
    from sklearn.cluster import KMeans
    from sklearn.neighbors import kneighbors_graph

    feature_vectors = np.asarray(feature_vectors)
    num_samples = len(feature_vectors)
    if feature_vectors.ndim != 2 or num_samples < 3:
        raise ValueError("feature_vectors must have shape [N, features], N >= 3")
    if not 1 <= num_neighbors < num_samples:
        raise ValueError("num_neighbors must be in [1, N - 1]")
    if max_clusters < 2:
        raise ValueError("max_clusters must be at least 2")

    # A k-NN relation is directed: j can be near i without i being near j.
    # Averaging A and A.T produces the symmetric graph required below.
    directed_affinity = kneighbors_graph(
        feature_vectors,
        n_neighbors=num_neighbors,
        mode="connectivity",
        include_self=False,
    )
    affinity = 0.5 * (directed_affinity + directed_affinity.T)
    normalized_laplacian = graph_laplacian(affinity, normed=True)

    # Dense eigh is transparent and fast for the 120-sample workshop case.
    # Large studies should keep the graph sparse and use a sparse eigensolver.
    eigenvalues, eigenvectors = np.linalg.eigh(normalized_laplacian.toarray())

    largest_candidate = min(max_clusters, num_samples - 1)
    candidate_counts = np.arange(2, largest_candidate + 1)
    # For candidate k, zero-based indexing makes the relevant jump
    # eigenvalues[k] - eigenvalues[k - 1].
    candidate_gaps = (
        eigenvalues[candidate_counts]
        - eigenvalues[candidate_counts - 1]
    )
    suggested_clusters = int(candidate_counts[np.argmax(candidate_gaps)])
    selected_clusters = suggested_clusters if num_clusters is None else num_clusters
    if not 2 <= selected_clusters <= largest_candidate:
        raise ValueError("num_clusters is outside the inspected candidate range")

    # Each graph node becomes one row in the first-k eigenvector coordinates.
    # Row normalization is the standard final preparation before k-means.
    spectral_coordinates = eigenvectors[:, :selected_clusters]
    row_norms = np.linalg.norm(spectral_coordinates, axis=1, keepdims=True)
    spectral_coordinates = spectral_coordinates / np.maximum(row_norms, 1e-12)
    labels = KMeans(
        n_clusters=selected_clusters,
        n_init=30,
        random_state=seed,
    ).fit_predict(spectral_coordinates)

    return {
        "labels": labels,
        "num_clusters": selected_clusters,
        "suggested_clusters": suggested_clusters,
        "affinity": affinity.tocsr(),
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "candidate_counts": candidate_counts,
        "candidate_gaps": candidate_gaps,
        "spectral_coordinates": spectral_coordinates,
    }


def run_kmeans_heatmap_baseline(feature_vectors, num_clusters, seed=7):
    """Cluster heatmap vectors directly with k-means for comparison."""
    from sklearn.cluster import KMeans

    estimator = KMeans(
        n_clusters=num_clusters,
        n_init=30,
        random_state=seed,
    )
    labels = estimator.fit_predict(feature_vectors)
    return {"labels": labels, "inertia": float(estimator.inertia_)}


def compute_tsne_embedding(feature_vectors, seed=7):
    """Project heatmap vectors to two dimensions for inspection only."""
    from sklearn.manifold import TSNE

    num_samples = len(feature_vectors)
    # Perplexity must be below N. The fixed seed and PCA initialization make
    # the teaching view repeatable in one software environment.
    perplexity = min(30.0, max(5.0, (num_samples - 1) / 3.0))
    perplexity = min(perplexity, num_samples - 1.0)
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    ).fit_transform(feature_vectors)
    return embedding, perplexity


def cluster_heatmaps(
    heatmaps,
    sample_records,
    output_size=20,
    normalize_each=True,
    num_neighbors=10,
    max_clusters=8,
    num_clusters=None,
    seed=7,
):
    """Run spectral clustering, direct k-means, and a shared t-SNE view."""
    from sklearn.metrics import adjusted_rand_score

    start = time.perf_counter()
    features, pooled_heatmaps = prepare_spray_features(
        heatmaps,
        output_size=output_size,
        normalize_each=normalize_each,
    )
    spectral = run_spray_spectral_clustering(
        features,
        num_neighbors=num_neighbors,
        max_clusters=max_clusters,
        num_clusters=num_clusters,
        seed=seed,
    )
    selected_clusters = spectral["num_clusters"]
    kmeans = run_kmeans_heatmap_baseline(features, selected_clusters, seed)
    embedding, perplexity = compute_tsne_embedding(features, seed)

    records_with_clusters = []
    for record, spectral_label, kmeans_label in zip(
        sample_records,
        spectral["labels"],
        kmeans["labels"],
    ):
        enriched_record = dict(record)
        enriched_record["spectral_cluster"] = int(spectral_label)
        enriched_record["kmeans_cluster"] = int(kmeans_label)
        records_with_clusters.append(enriched_record)

    watermark_labels = np.asarray(
        [record["imagenet_w_watermark"] for record in sample_records],
        dtype=int,
    )
    metrics = {
        "spectral_vs_watermark": adjusted_rand_score(
            watermark_labels,
            spectral["labels"],
        ),
        "kmeans_vs_watermark": adjusted_rand_score(
            watermark_labels,
            kmeans["labels"],
        ),
        "method_agreement": adjusted_rand_score(
            spectral["labels"],
            kmeans["labels"],
        ),
    }
    return {
        "features": features,
        "pooled_heatmaps": pooled_heatmaps,
        "spectral": spectral,
        "spectral_labels": spectral["labels"],
        "kmeans_labels": kmeans["labels"],
        "embedding": embedding,
        "perplexity": perplexity,
        "metrics": metrics,
        "sample_records": records_with_clusters,
        "seconds": time.perf_counter() - start,
    }


def summarize_watermark_clusters(cluster_labels, sample_records):
    """Evaluate cluster enrichment after revealing the held-out watermark labels.

    The labels are used only after clustering. ``candidate_cluster`` is the
    cluster with the largest watermark fraction and should be treated as a
    candidate for inspection, not as an automatically validated explanation.
    """
    cluster_labels = np.asarray(cluster_labels)
    if len(cluster_labels) != len(sample_records):
        raise ValueError("cluster_labels and sample_records must have equal length")

    watermark_labels = np.asarray(
        [record["imagenet_w_watermark"] for record in sample_records],
        dtype=bool,
    )
    total_watermarked = int(watermark_labels.sum())
    cluster_rows = []

    for cluster_index in np.unique(cluster_labels):
        in_cluster = cluster_labels == cluster_index
        cluster_size = int(in_cluster.sum())
        watermarked_in_cluster = int(watermark_labels[in_cluster].sum())
        cluster_rows.append(
            {
                "cluster": int(cluster_index),
                "samples": cluster_size,
                "watermarked": watermarked_in_cluster,
                "watermark_rate": watermarked_in_cluster / cluster_size,
            }
        )

    candidate = max(cluster_rows, key=lambda row: row["watermark_rate"])
    base_rate = total_watermarked / len(sample_records)
    recall = (
        candidate["watermarked"] / total_watermarked
        if total_watermarked
        else float("nan")
    )
    enrichment = (
        candidate["watermark_rate"] / base_rate
        if base_rate
        else float("nan")
    )
    return {
        "clusters": cluster_rows,
        "candidate_cluster": candidate["cluster"],
        "base_rate": base_rate,
        "candidate_precision": candidate["watermark_rate"],
        "candidate_recall": recall,
        "candidate_enrichment": enrichment,
    }


def build_spectral_diagnostics_figure(spectral_result):
    """Create a compact Plotly view of small eigenvalues and eigengaps."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    eigenvalues = spectral_result["eigenvalues"]
    candidate_counts = spectral_result["candidate_counts"]
    candidate_gaps = spectral_result["candidate_gaps"]
    num_clusters = spectral_result["num_clusters"]
    num_values = min(12, len(eigenvalues))

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Small graph-Laplacian eigenvalues", "Eigengap heuristic"),
    )
    figure.add_trace(
        go.Scatter(
            x=np.arange(num_values),
            y=eigenvalues[:num_values],
            mode="lines+markers",
            name="eigenvalues",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(x=candidate_counts, y=candidate_gaps, name="eigengaps"),
        row=1,
        col=2,
    )
    figure.add_vline(
        x=num_clusters - 0.5,
        line_dash="dash",
        line_color="firebrick",
        row=1,
        col=1,
    )
    figure.add_vline(
        x=num_clusters,
        line_dash="dash",
        line_color="firebrick",
        row=1,
        col=2,
    )
    figure.update_xaxes(title_text="eigenvalue index", row=1, col=1)
    figure.update_xaxes(title_text="candidate clusters k", row=1, col=2)
    figure.update_yaxes(title_text="eigenvalue", row=1, col=1)
    figure.update_yaxes(title_text="lambda_k - lambda_(k-1)", row=1, col=2)
    figure.update_layout(
        template="plotly_white",
        height=430,
        showlegend=False,
        title="SpRAy spectral diagnostics",
    )
    return figure


def build_cluster_comparison_figure(embedding, clustering_labels):
    """Compare several cluster assignments on one shared 2D embedding."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    embedding = np.asarray(embedding)
    if embedding.ndim != 2 or embedding.shape[1] != 2:
        raise ValueError("embedding must have shape [samples, 2]")
    if not clustering_labels:
        raise ValueError("clustering_labels must contain at least one method")

    method_names = list(clustering_labels)
    figure = make_subplots(
        rows=1,
        cols=len(method_names),
        subplot_titles=method_names,
        horizontal_spacing=0.08,
    )
    sample_indices = np.arange(len(embedding))
    for column, method_name in enumerate(method_names, start=1):
        labels = np.asarray(clustering_labels[method_name])
        if labels.shape != (len(embedding),):
            raise ValueError(f"Labels for {method_name!r} have the wrong shape")
        figure.add_trace(
            go.Scattergl(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers",
                customdata=sample_indices,
                marker={
                    "color": labels,
                    "colorscale": "Turbo",
                    "size": 9,
                    "opacity": 0.82,
                },
                text=[f"cluster {int(label)}" for label in labels],
                hovertemplate="sample %{customdata}<br>%{text}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(title_text="t-SNE 1", row=1, col=column)
        figure.update_yaxes(title_text="t-SNE 2", row=1, col=column)

    figure.update_layout(
        template="plotly_white",
        height=500,
        title="Same heatmaps and t-SNE coordinates, different clustering rules",
    )
    return figure


def _normalized_bwr_rgb(heatmap, percentile=99.5):
    """Map a signed heatmap to RGB using symmetric percentile clipping."""
    from matplotlib import colormaps

    heatmap = np.asarray(heatmap, dtype=np.float32)
    scale = float(np.percentile(np.abs(heatmap), percentile))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    normalized = np.clip(heatmap / scale, -1.0, 1.0)
    rgb = colormaps["bwr"]((normalized + 1.0) / 2.0)[..., :3]
    return rgb, normalized


def _sample_panel_data_uri(sample_record, heatmap, thumbnail_size=180):
    """Encode input, signed heatmap, and relevance overlay as one JPEG URI."""
    display_tensor = load_display_image(sample_record["path"])[0]
    if sample_record.get("imagenet_w_watermark", False):
        display_tensor = add_imagenet_w_watermark(display_tensor)
    image_rgb = display_tensor.permute(1, 2, 0).numpy()
    heatmap_rgb, normalized_heatmap = _normalized_bwr_rgb(heatmap)

    relevance_alpha = 0.72 * np.abs(normalized_heatmap)[..., None] ** 0.7
    overlay_rgb = (1.0 - relevance_alpha) * image_rgb + relevance_alpha * heatmap_rgb

    panels = []
    for panel_rgb in (image_rgb, heatmap_rgb, overlay_rgb):
        panel = Image.fromarray(np.uint8(np.clip(panel_rgb, 0.0, 1.0) * 255))
        panel = panel.resize((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
        panels.append(np.asarray(panel))

    separator = np.full((thumbnail_size, 4, 3), 255, dtype=np.uint8)
    combined = np.concatenate(
        [panels[0], separator, panels[1], separator, panels[2]],
        axis=1,
    )
    output_buffer = io.BytesIO()
    Image.fromarray(combined).save(output_buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(output_buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_clickable_spray_explorer(
    embedding,
    cluster_labels,
    sample_records,
    heatmaps,
    output_path=None,
    include_plotlyjs=True,
):
    """Build a self-contained Plotly scatter whose points open sample panels.

    The returned HTML works without ``FigureWidget`` or a live Python callback.
    If ``output_path`` is supplied, the same HTML is saved as a portable file.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError("Install plotly to build the SpRAy explorer") from error

    embedding = np.asarray(embedding)
    cluster_labels = np.asarray(cluster_labels)
    heatmaps = torch.as_tensor(heatmaps).detach().cpu().numpy()
    num_samples = len(sample_records)
    if embedding.shape != (num_samples, 2):
        raise ValueError("embedding must have shape [num_samples, 2]")
    if cluster_labels.shape != (num_samples,):
        raise ValueError("cluster_labels must have shape [num_samples]")
    if heatmaps.shape[0] != num_samples:
        raise ValueError("heatmaps and sample_records must have equal length")

    browser_samples = []
    for sample_index, (record, heatmap) in enumerate(zip(sample_records, heatmaps)):
        metadata_lines = [
            f"Cluster: {int(cluster_labels[sample_index])}",
            f"Dataset sample: {record['dataset_index']}",
            f"Target: {record['target_name']}",
            (
                f"Prediction: {record['predicted_name']} "
                f"({record['predicted_probability']:.1%})"
            ),
            f"Target logit: {record['target_logit']:+.3f}",
            f"File: {record['path']}",
        ]
        if "imagenet_w_watermark" in record:
            watermark_status = (
                "present" if record["imagenet_w_watermark"] else "absent"
            )
            metadata_lines.insert(2, f"ImageNet-W watermark: {watermark_status}")
        if "kmeans_cluster" in record:
            metadata_lines.insert(1, f"Direct k-means cluster: {record['kmeans_cluster']}")
        browser_samples.append(
            {
                "panel": _sample_panel_data_uri(record, heatmap),
                "metadata": "\n".join(metadata_lines),
            }
        )

    figure = go.Figure()
    for cluster_index in sorted(np.unique(cluster_labels)):
        sample_indices = np.flatnonzero(cluster_labels == cluster_index)
        hover_text = []
        for sample_index in sample_indices:
            record = sample_records[int(sample_index)]
            hover_text.append(
                "<br>".join(
                    [
                        f"sample {record['dataset_index']}",
                        f"prediction: {record['predicted_name']}",
                        f"confidence: {record['predicted_probability']:.1%}",
                        f"target logit: {record['target_logit']:+.3f}",
                    ]
                )
            )

        figure.add_trace(
            go.Scatter(
                x=embedding[sample_indices, 0],
                y=embedding[sample_indices, 1],
                mode="markers",
                name=f"cluster {int(cluster_index)} (n={len(sample_indices)})",
                customdata=sample_indices,
                hovertext=hover_text,
                hovertemplate="%{hovertext}<extra></extra>",
                marker={"size": 9, "opacity": 0.82, "line": {"width": 0.5}},
            )
        )

    figure.update_layout(
        title="SpRAy clusters — click a point to inspect its explanation",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        clickmode="event+select",
        template="plotly_white",
        height=610,
        legend={"orientation": "h", "y": -0.14},
        margin={"l": 50, "r": 20, "t": 65, "b": 95},
    )

    samples_json = json.dumps(browser_samples).replace("</", "<\\/")
    post_script = f"""
(function() {{
    const plot = document.getElementById('{{plot_id}}');
    const samples = {samples_json};
    const inspector = document.createElement('div');
    inspector.style.cssText = 'margin:16px 4px;padding:14px;border:1px solid #ddd;border-radius:8px;background:#fafafa;';
    inspector.innerHTML = `
      <div style="font-weight:600;margin-bottom:8px">Selected explanation</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;max-width:548px;text-align:center;font-size:12px;margin-bottom:4px">
        <span>model input</span><span>signed LRP</span><span>overlay</span>
      </div>
      <img id="{{plot_id}}-spray-panel" style="width:min(100%,548px);height:auto;border:1px solid #eee" />
      <pre id="{{plot_id}}-spray-metadata" style="white-space:pre-wrap;margin:10px 0 0;font:13px/1.45 monospace"></pre>`;
    plot.parentNode.appendChild(inspector);

    function showSample(sampleIndex) {{
        const sample = samples[Number(sampleIndex)];
        document.getElementById('{{plot_id}}-spray-panel').src = sample.panel;
        document.getElementById('{{plot_id}}-spray-metadata').textContent = sample.metadata;
    }}

    plot.on('plotly_click', function(event) {{
        showSample(event.points[0].customdata);
    }});
    showSample(0);
}})();
"""
    html = figure.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        post_script=post_script,
        config={"displaylogo": False, "responsive": True},
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
    return html
