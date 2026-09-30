# dashAI Vision-Language plugin

[![PyPI](https://img.shields.io/pypi/v/dashai-vision-language-plugin)](https://pypi.org/project/dashai-vision-language-plugin/)
[![Python](https://img.shields.io/pypi/pyversions/dashai-vision-language-plugin)](https://pypi.org/project/dashai-vision-language-plugin/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Zero-shot image-classification plugin for [dashAI](https://github.com/DashAISoftware/dashAI). Classify images into any set of labels you choose, with no training or fine-tuning, using pretrained Hugging Face vision-language models.

It adds seven components to dashAI's image classification task, all Zero-Shot: **CLIP** (three sizes), **SigLIP**, **ALIGN**, **AltCLIP**, and **MetaCLIP 2**.

## Installation

### From dashAI (recommended)

Open the **Plugins** page in dashAI, search for `dashai-vision-language-plugin`, and install it with one click. No restart is needed.

### With pip

Install it in the same Python environment where dashAI runs:

```bash
pip install dashai-vision-language-plugin
```

Restart dashAI afterwards. It discovers the plugin automatically through the `dashai.plugins` entry point.

### Requirements

- Python 3.10 or newer
- dashAI 0.9.7.post2 or newer
- Internet access the first time you use a model, to download its checkpoint from the Hugging Face Hub (see [System Requirements](#system-requirements))

Training downloads the selected Hugging Face checkpoint unless it is already present in the local Hugging Face cache. After loading a saved checkpoint (see [Persistence](#persistence)), the checkpoint is instead downloaded lazily before the first prediction.

## Use in dashAI

1. Create or load an image dataset and select its image column.
2. Create an `ImageClassificationTask` with the dataset's categorical label column.
3. In the task's component selector, choose one of the Zero-Shot components (a CLIP size, SigLIP, ALIGN, AltCLIP, or MetaCLIP 2).
4. Configure the model, prompt template, batch size, and device.
5. Run the task, then evaluate its predictions with the normal dashAI evaluation flow.

The labels in the training dataset define the zero-shot classes. For labels `cat`, `dog`, and `car`, the default template produces the prompts `a photo of a cat`, `a photo of a dog`, and `a photo of a car`.

**`train()` does not fine-tune the backbone.** It records the dataset class labels and computes their text embeddings; image inference compares each image embedding with those prompts.

## Components

### CLIP Zero-Shot

Unlike the other components below, the CLIP checkpoint is **not** a free-text parameter. Following dashAI's own convention for same-family, different-size models (e.g. `ResNet18ImageClassifier`/`ResNet50ImageClassifier`), each CLIP size is its own component with the Hugging Face checkpoint fixed in code, so users can pick a named option instead of typing a model ID:

| Component | Fixed checkpoint | Trade-off |
| --- | --- | --- |
| `CLIPViTB32ZeroShotClassifier` | `openai/clip-vit-base-patch32` | Fastest, smallest; the default choice. |
| `CLIPViTB16ZeroShotClassifier` | `openai/clip-vit-base-patch16` | Mid-sized: more accurate than B/32, lighter than L/14. |
| `CLIPViTL14ZeroShotClassifier` | `openai/clip-vit-large-patch14` | Largest, most accurate, slowest, heaviest download. |

All three share the same other parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `prompt_template` | `a photo of a {}` | Text prompt with exactly one `{}` label placeholder. |
| `batch_size` | `32` | Number of images processed per inference batch; must be at least 1. |
| `device` | `auto` | `auto` uses CUDA when available, otherwise CPU; `cpu` forces CPU; `cuda` requires CUDA to be available. |

They score each image-label pair with CLIP's joint softmax over labels (`softmax(logit_scale.exp() * cosine_similarity)`), so the reported probabilities are a standard categorical distribution. Need a different CLIP checkpoint than these three (e.g. a fine-tuned or community variant)? Subclass `CLIPZeroShotClassifier` and set `MODEL_NAME`; that's exactly what these three components do.

### SigLIP Zero-Shot (`SigLIPZeroShotClassifier`)

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model_name` | `google/siglip-base-patch16-224` | Hugging Face SigLIP checkpoint ID. |
| `prompt_template` | `a photo of a {}` | Text prompt with exactly one `{}` label placeholder. |
| `batch_size` | `32` | Number of images processed together during inference; must be at least 1. |
| `device` | `auto` | `auto` uses CUDA when available, otherwise CPU; `cpu` forces CPU; `cuda` requires CUDA to be available. |

SigLIP was trained with an independent sigmoid loss per image-label pair (`sigmoid(logit_scale.exp() * cosine_similarity + logit_bias)`), not a joint softmax, so out of the box its per-label scores don't sum to 1 across labels. To stay compatible with dashAI's classification metrics (which expect a per-sample categorical distribution), this component renormalizes the sigmoid scores to sum to 1. This changes only the reported non-argmax probabilities, not the predicted class.

**SigLIP2:** the fixed-resolution SigLIP2 checkpoints (e.g. `google/siglip2-base-patch16-224`) declare `model_type: "siglip"` in their config and load correctly through this same `SigLIPZeroShotClassifier` component; just set `model_name` to a SigLIP2 checkpoint ID, no separate component needed. Only the variable resolution "NaFlex" SigLIP2 checkpoints require the newer `Siglip2Model`/`Siglip2Processor` classes with different image handling (`pixel_attention_mask`, `spatial_shapes`), which this plugin does not implement.

### ALIGN Zero-Shot (`ALIGNZeroShotClassifier`)

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model_name` | `kakaobrain/align-base` | Hugging Face ALIGN checkpoint ID. |
| `prompt_template` | `a photo of a {}` | Text prompt with exactly one `{}` label placeholder. |
| `batch_size` | `32` | Number of images processed together during inference; must be at least 1. |
| `device` | `auto` | `auto` uses CUDA when available, otherwise CPU; `cpu` forces CPU; `cuda` requires CUDA to be available. |

ALIGN pairs an EfficientNet vision encoder with a BERT text encoder. Unlike CLIP's `logit_scale.exp()` multiplier, ALIGN divides cosine similarity by a learned scalar `temperature` before a joint softmax over labels (`softmax(cosine_similarity / temperature)`), so its reported probabilities are already a standard categorical distribution.

### AltCLIP Zero-Shot (`AltCLIPZeroShotClassifier`)

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model_name` | `BAAI/AltCLIP` | Hugging Face AltCLIP checkpoint ID. |
| `prompt_template` | `a photo of a {}` | Text prompt with exactly one `{}` label placeholder. |
| `batch_size` | `32` | Number of images processed together during inference; must be at least 1. |
| `device` | `auto` | `auto` uses CUDA when available, otherwise CPU; `cpu` forces CPU; `cuda` requires CUDA to be available. |

AltCLIP swaps CLIP's text encoder for a multilingual XLM-R encoder while keeping CLIP's exact joint softmax scoring, making it a good fit for class labels or prompts written in languages other than English. It is a larger checkpoint than the others (CLIP ViT-L/14 + XLM-R-large).

### MetaCLIP 2 Zero-Shot (`MetaCLIP2ZeroShotClassifier`)

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model_name` | `facebook/metaclip-2-worldwide-s16` | Hugging Face MetaCLIP 2 checkpoint ID. |
| `prompt_template` | `a photo of a {}` | Text prompt with exactly one `{}` label placeholder. |
| `batch_size` | `32` | Number of images processed together during inference; must be at least 1. |
| `device` | `auto` | `auto` uses CUDA when available, otherwise CPU; `cpu` forces CPU; `cuda` requires CUDA to be available. |

MetaCLIP 2 uses the same joint softmax scoring and processor as CLIP, but is trained on 300+ languages, so it is a multilingual alternative to AltCLIP. The default checkpoint is the smallest published MetaCLIP 2 variant (ViT-S/16); larger `facebook/metaclip-2-worldwide-*` checkpoints are available for higher accuracy at a larger download/RAM cost.

## System Requirements

Measured against `CLIPViTB32ZeroShotClassifier`'s checkpoint (`openai/clip-vit-base-patch32`). SigLIP, ALIGN, and MetaCLIP 2's default checkpoints are similar in size or smaller; **`CLIPViTL14ZeroShotClassifier` and AltCLIP's checkpoint (`BAAI/AltCLIP`) are notably larger**, so budget more disk and RAM if you use them.

| Resource | Requirement |
| --- | --- |
| Disk | ~2GB total for one default checkpoint: ~700MB for `torch` + `transformers` (already included in dashAI), plus ~1.1GB cached. Each additional cached checkpoint adds its own size; AltCLIP alone can add several GB. |
| RAM / CPU | Runs comfortably on CPU with 4-8GB RAM for the small default checkpoints (~150-200M parameters); AltCLIP needs more. |
| GPU | Optional. `device="cuda"`/`"auto"` use CUDA when available (including ROCm builds of PyTorch, which expose the same CUDA API). **Apple Silicon (MPS) is not supported**; this matches dashAI's own models, which only distinguish CUDA vs. CPU. |
| Network | Required on first use to download the checkpoint from the Hugging Face Hub, unless it is already cached locally. |
| Software | Python >=3.10, dashAI >=0.9.7.post2 (see `pyproject.toml`). |

## Persistence

Saved checkpoints are intentionally lightweight: they keep component configuration and class labels, not the backbone's model weights. Loading a checkpoint requires the named Hugging Face checkpoint to be available again; it will be re-downloaded when absent from the local cache.

## Contributing

Issues and pull requests are welcome at [github.com/diegoolguinw/dashai_vision_language_plugin](https://github.com/diegoolguinw/dashai_vision_language_plugin).

Set up a development environment:

```bash
git clone https://github.com/diegoolguinw/dashai_vision_language_plugin.git
cd dashai_vision_language_plugin
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e '.[dev]'
```

Run the ordinary unit suite (the real-model integration test is excluded by default):

```bash
python -m pytest
```

Lint the project:

```bash
python -m ruff check .
```

Build source and wheel distributions:

```bash
python -m build
```

Run the opt-in smoke tests. This downloads and executes all seven real checkpoints if needed (including the larger CLIP ViT-L/14 and AltCLIP checkpoints):

```bash
RUN_CLIP_INTEGRATION=1 python -m pytest -m integration -v
```

## Author

Diego Olguin-Wende - dolguin at dim dot uchile dot cl

Bug reports and questions: [GitHub Issues](https://github.com/diegoolguinw/dashai_vision_language_plugin/issues).

## License

Released under the [MIT License](LICENSE).
