# dashAI CLIP Model Package

A zero-shot image-classification plugin for [dashAI](https://github.com/DashboardAI/dashAI), powered by CLIP.

## Installation

For a published release:

```bash
pip install dashai-clip-model-package
```

For local development:

```bash
pip install -e '.[dev]'
```

Training downloads the selected Hugging Face CLIP checkpoint unless it is already present in the local Hugging Face cache. After loading a saved checkpoint (see [Persistence](#persistence)), the checkpoint is instead downloaded lazily before the first prediction.

## Use in dashAI

1. Create or load an image dataset and select its image column.
2. Create an `ImageClassificationTask` with the dataset's categorical label column.
3. In the task's component selector, choose **CLIP Zero-Shot**.
4. Configure the model, prompt template, batch size, and device.
5. Run the task, then evaluate its predictions with the normal dashAI evaluation flow.

The labels in the training dataset define the zero-shot classes. For labels `cat`, `dog`, and `car`, the default template produces the prompts `a photo of a cat`, `a photo of a dog`, and `a photo of a car`.

**`train()` does not fine-tune CLIP.** It records the dataset class labels and computes their text embeddings; image inference compares each image embedding with those prompts.

## Configuration

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model_name` | `openai/clip-vit-base-patch32` | Hugging Face CLIP checkpoint ID. |
| `prompt_template` | `a photo of a {}` | Text prompt with exactly one `{}` label placeholder. |
| `batch_size` | `32` | Number of images processed per inference batch; must be at least 1. |
| `device` | `auto` | `auto` uses CUDA when available, otherwise CPU; `cpu` forces CPU; `cuda` requires CUDA to be available. |

## Persistence

Saved checkpoints are intentionally lightweight: they keep component configuration and class labels, not the CLIP model weights. Loading a checkpoint requires the named Hugging Face checkpoint to be available again; it will be re-downloaded when absent from the local cache.

## Development checks

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

Run the opt-in smoke test. This downloads and executes the real checkpoint if needed:

```bash
RUN_CLIP_INTEGRATION=1 python -m pytest -m integration -v
```
