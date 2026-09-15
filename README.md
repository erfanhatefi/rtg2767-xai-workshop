# RTG 2767 XAI workshop

Four guided notebooks for a three-hour workshop on attribution methods:

1. perturbation and image SHAP;
2. gradients, SmoothGrad, CAM, and Grad-CAM;
3. LRP and Concept Relevance Propagation;
4. automated explanation analysis with SpRAy.

The shared case study uses ImageNet-W to add a controlled watermark to
Imagenette French-horn images. Explanations compare the `carton` and
`French horn` targets; attribution is not presented as proof of causality.

## Google Colab

Open a notebook through GitHub or upload it to Colab, then run its collapsed
environment cell. The cell clones this repository and installs the additional
dependencies. A network connection is still required for Imagenette and model
weights.

Before distributing links, confirm that `https://github.com/erfanhatefi/rtg2767-xai-workshop.git` is public.
After creating a frozen workshop tag, change `PUBLIC_REPOSITORY_REF` in the
notebooks from `main` to that tag.

## Local execution

Use Python with PyTorch and torchvision installed, then run:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

Open the notebooks from `notebooks/`. Generated data and plots are written to
`workshop_data/` and `workshop_outputs/`.

## Repository layout

- `notebooks/` contains the complete presenter notebooks.
- `utils/` contains workshop-facing functions with short, explicit names.
- `lexplainet/implicit/` contains the implicit LRP implementation.
- `lexplainet/heatmap/` contains attribution visualization utilities.
- `workshop_cache/` contains the prepared 120-image SpRAy heatmap cache.

The cache stores explanations, not the original images. Imagenette is still
downloaded so the interactive SpRAy view can display each source image. Cache
metadata verifies the model, target, ordered dataset indices, and watermark
assignments before reuse.

## Teaching recommendation

Use these as guided notebooks rather than fill-in-the-blank exercises. Hide the
setup cells, explain one representative implementation per notebook, and let
participants change targets, patch sizes, channels, or clustering settings.
Run Notebook 4 from the prepared cache rather than extracting all heatmaps live.

## Data and references

- Lapuschkin et al., *Unmasking Clever Hans Predictors and Assessing What
  Machines Really Learn*, Nature Communications (2019).
- ImageNet-W from the Whac-A-Mole project.
- Imagenette from fastai.

Add an explicit software license and any required third-party attribution
before publishing this repository.
