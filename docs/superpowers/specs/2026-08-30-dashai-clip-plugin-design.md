# dashAI CLIP Zero-Shot Plugin Design

## Goal

Build `dashai-clip-model-package`, a self-contained Python plugin that adds
CLIP-based zero-shot image classification to dashAI without changing dashAI
core. The MVP targets dashAI `0.9.7.post2` as observed on the `production`
branch at commit `f37f6d2d032203378c04c9da4930b055f2773817` (2026-08-12).

The plugin must be discoverable through the `dashai.plugins` entry-point group,
appear as compatible with `ImageClassificationTask`, and return class
probabilities in dashAI's expected ordering.

## Scope

The MVP includes:

- Hugging Face `CLIPModel` and `CLIPProcessor` integration.
- Zero-shot image classification against the dataset's class labels.
- Configurable checkpoint, prompt template, batch size, and device.
- CPU inference and CUDA inference when available.
- Lightweight persistence of configuration and class metadata.
- Unit, packaging, entry-point, and optional real-model smoke tests.
- Installation and usage documentation.

The MVP excludes fine-tuning, retrieval, embedding converters, new task or
vision-language base classes, alternative backends, prompt ensembling, and
changes to dashAI core.

## Package Architecture

The repository will contain a standard `src/`-layout Python package:

```text
dashai_multimodal_plugin/
├── LICENSE
├── README.md
├── pyproject.toml
├── src/
│   └── dashai_clip_model_package/
│       ├── __init__.py
│       └── clip_zero_shot_classifier.py
└── tests/
    ├── test_clip_zero_shot_classifier.py
    ├── test_entry_point.py
    └── test_real_clip.py
```

The public component is `CLIPZeroShotClassifier`. It inherits directly from
`DashAI.back.models.base_model.BaseModel`, declares
`COMPATIBLE_COMPONENTS = ["ImageClassificationTask"]`, and exposes a dashAI
`BaseSchema` through `SCHEMA`.

Direct inheritance from `BaseModel` is intentional. The current
`TorchvisionImageClassifier` base class assumes a trainable supervised
backbone, optimizer, epochs, and a classifier head; those contracts do not fit
zero-shot CLIP inference.

## Configuration Schema

The component exposes four GUI-facing parameters:

- `model_name: str`, default `openai/clip-vit-base-patch32`.
- `prompt_template: str`, default `a photo of a {}`.
- `batch_size: int`, default `32`, constrained to values greater than or equal
  to one.
- `device: str`, one of `auto`, `cpu`, or `cuda`, default `auto`.

The prompt convention accepts exactly one positional `{}` replacement field.
Named fields, missing fields, conversions, format specifications, and multiple
replacement fields are rejected with a configuration error. This keeps prompt
generation deterministic and avoids accepting templates that fail only during
training.

Schema labels and descriptions use dashAI's `MultilingualString`. English and
Spanish text are supplied for the MVP, with the same safe English fallback for
the other locales expected by the current dashAI schema conventions if the
runtime requires every locale.

## Runtime State and Loading

Construction validates scalar configuration and resolves the requested device,
but does not download or instantiate Hugging Face objects. Model and processor
loading is lazy and happens on the first operation that needs them. This makes
component discovery fast, prevents network activity during imports, and allows
default tests to replace the backend.

Runtime state consists of:

- configured model identifier, prompt template, batch size, and device choice;
- resolved `torch.device`;
- ordered `class_names`, `label_to_idx`, and `idx_to_label` mappings;
- lazily loaded processor and model;
- cached normalized text features for the current class ordering.

Changing class metadata invalidates the cached text features.

## Class Ordering and Training Semantics

`train(x_train, y_train, x_validation=None, y_validation=None)` performs no
gradient updates and never changes CLIP parameters. It:

1. Identifies the single output column.
2. Reads its dashAI `Categorical.categories` ordering when present.
3. Falls back to first-seen unique target values only when canonical category
   metadata is unavailable.
4. Stores the resulting class mappings.
5. Loads the backend and prepares normalized text embeddings.
6. Returns `self`.

The canonical category order is never independently sorted. Prompt order,
embedding order, output columns, `prepare_output()`, and persisted mappings all
use the same class sequence.

`prepare_output()` maps target strings to integer indices in that sequence so
dashAI classification metrics compare targets and probability columns
consistently.

## Prompt Generation

For each class label, the plugin converts the value to text, replaces
underscores with spaces, and inserts the result into the prompt template.
Other spelling, capitalization, and punctuation are preserved.

For example:

```text
golden_retriever -> golden retriever -> a photo of a golden retriever
```

Prompt construction is a pure deterministic helper and is tested separately.

## Prediction Data Flow

`predict(x)` requires class metadata established by `train()` or restored by
`load()`. It then:

1. Locates the single image column.
2. Reads each dashAI image value and calls `DashAIImage.to_pil()`.
3. Converts images to RGB before handing them to `CLIPProcessor`.
4. Processes images in stable, non-shuffled batches.
5. Runs inference under `torch.inference_mode()` with the model in evaluation
   mode.
6. Normalizes image features and computes their matrix product with cached
   normalized text features.
7. Multiplies similarities by `model.logit_scale.exp()`.
8. Applies softmax over classes and returns a NumPy `float32` array with shape
   `(n_samples, n_classes)`.

An empty input returns an empty array with shape `(0, n_classes)`. Output rows
remain in dataset order and output columns remain in canonical class order.

## Device Behavior

- `auto` selects CUDA when `torch.cuda.is_available()` is true and otherwise
  selects CPU.
- `cpu` always selects CPU.
- `cuda` selects CUDA only when available; otherwise construction raises a
  clear `RuntimeError` explaining how to use `auto` or `cpu`.

Inputs, cached text features, and the model are placed on the same resolved
device. Native CUDA out-of-memory errors are allowed to propagate with a short
batch-size hint added while preserving the original exception as the cause.

## Persistence

`save(filename)` writes a small versioned checkpoint containing configuration,
class names, and mappings. It does not serialize model weights, processor
objects, text embeddings, or optimizer state.

`load(filename)` is a class method that:

1. Loads the checkpoint on CPU.
2. Validates its format version and required keys.
3. Reconstructs the component with the saved configuration.
4. Restores class ordering and mappings.
5. Leaves the Hugging Face backend unloaded until prediction.

This makes artifacts portable but means loading for inference requires the
configured Hugging Face checkpoint to be available from cache or network.

## Error Handling

The component raises focused errors for:

- invalid prompt templates;
- unsupported device names or unavailable requested CUDA;
- missing or empty class metadata;
- an unexpected number of input or output columns;
- values that do not provide dashAI image decoding behavior;
- malformed or unsupported saved checkpoints;
- Hugging Face model or processor loading failures.

Backend-loading errors include the model identifier. Image-conversion errors
include the sample index without exposing image bytes. No import-time network
calls occur.

## Packaging and Metadata

`pyproject.toml` uses the current dashAI-compatible Python floor (`>=3.10`), a
modern build backend, and distribution name `dashai-clip-model-package`. The
import package is `dashai_clip_model_package`.

The entry point is:

```toml
[project.entry-points.'dashai.plugins']
CLIPZeroShotClassifier = 'dashai_clip_model_package.clip_zero_shot_classifier:CLIPZeroShotClassifier'
```

Metadata includes keywords `DashAI` and `Model`, an MIT license, and:

- Author: Diego Olguín-Wende
- Email: dolguin@dim.uchile.cl

Runtime dependencies are limited to packages the component imports directly,
with compatible lower bounds rather than unnecessarily strict pins. dashAI,
Transformers, PyTorch, NumPy, and Pillow are represented according to the
resolved packaging contract; dependencies already installed by dashAI are not
vendored.

## Testing Strategy

Default tests must not download model weights.

Unit tests use small fake processor/model objects and synthetic embeddings to
cover:

- prompt validation and deterministic construction;
- underscore normalization;
- device resolution;
- canonical and fallback class ordering;
- no-gradient/no-training behavior;
- probability shape, non-negativity, row sums, and class order;
- ordered batching when sample count exceeds batch size;
- empty prediction input;
- output-label preparation;
- save/load round trips without serializing backend objects;
- contextual errors for invalid images and backend loading.

A packaging test builds and installs the distribution into an isolated
environment, then checks the `dashai.plugins` entry point through
`importlib.metadata`.

An optional test marked `integration` exercises a real compatible CLIP
checkpoint. It is excluded from the normal test command and documented with an
explicit command and network/cache requirements.

Verification follows repository configuration and includes, at minimum:

```text
pytest
ruff check .
ruff format --check .
python -m build
```

The final verification also imports the installed package and loads its entry
point in an environment compatible with the inspected dashAI production
version.

## Documentation

The README explains installation from the repository and from PyPI after
publication, the dashAI GUI workflow, all four settings, CPU/CUDA behavior, a
three-class example, persistence behavior, and how to run both default and
real-model tests. It states prominently that `train()` only establishes class
prompts and that the plugin performs zero-shot inference rather than supervised
fine-tuning.

## Acceptance Criteria

The implementation is accepted when:

- the package builds and installs cleanly;
- dashAI discovers the component via its entry point;
- the component is compatible with `ImageClassificationTask`;
- training records the correct ordered classes without changing model weights;
- CPU and automatic device selection work as specified;
- CUDA is supported and unavailable CUDA requests fail clearly;
- normal dashAI image values are decoded without external conversion;
- prediction returns a valid `(n_samples, n_classes)` probability matrix in
  canonical class order;
- lightweight save/load reconstructs a usable component;
- default tests and configured quality checks pass without downloading CLIP;
- the optional real-model smoke test is documented and runnable; and
- no dashAI core changes are required.
