# dashAI CLIP Zero-Shot Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an installable dashAI plugin that performs CLIP zero-shot image classification and returns correctly ordered class-probability matrices.

**Architecture:** Pure helpers validate prompts and resolve devices; a single `CLIPZeroShotClassifier` adapts dashAI datasets to a lazily loaded Hugging Face CLIP backend. The component persists only configuration and class metadata, so normal tests use fakes and never download model weights.

**Tech Stack:** Python 3.10+, dashAI 0.9.7.post2, PyTorch, Hugging Face Transformers, Pillow, NumPy, pytest, Ruff, hatchling/build.

**Spec:** `docs/superpowers/specs/2026-08-30-dashai-clip-plugin-design.md`

## Global Constraints

- Distribution name: `dashai-clip-model-package`; import package: `dashai_clip_model_package`.
- Public component: `CLIPZeroShotClassifier`.
- Entry-point group: `dashai.plugins`.
- Compatible task: `ImageClassificationTask`; do not modify dashAI core.
- Default checkpoint: `openai/clip-vit-base-patch32`.
- Default prompt: `a photo of a {}`; accept exactly one plain positional field.
- Devices: `auto`, `cpu`, `cuda`; default `auto`.
- Prediction output: NumPy `float32`, shape `(n_samples, n_classes)`, canonical class order.
- No import-time downloads and no default-test model downloads.
- Persist configuration and class metadata, never pretrained CLIP weights.
- Author: Diego Olguín-Wende `<dolguin@dim.uchile.cl>`; license: MIT.
- This directory currently has no Git metadata. Run commit steps only after the user initializes or supplies a Git repository; otherwise record the task checkpoint in the plan without creating a repository implicitly.

## File Map

- `pyproject.toml`: build metadata, runtime/dev dependencies, Ruff/pytest settings, and dashAI entry point.
- `README.md`: initially supplies build metadata, then expands into complete user documentation.
- `src/dashai_clip_model_package/__init__.py`: stable public export.
- `src/dashai_clip_model_package/prompts.py`: prompt-template validation, label normalization, prompt construction.
- `src/dashai_clip_model_package/device.py`: deterministic CPU/CUDA resolution.
- `src/dashai_clip_model_package/clip_zero_shot_classifier.py`: dashAI schema, lifecycle, dataset adaptation, CLIP inference, and persistence.
- `tests/conftest.py`: lightweight dataset, image, processor, and CLIP fakes shared by unit tests.
- `tests/test_prompts.py`: prompt validation and construction.
- `tests/test_device.py`: device selection and error behavior.
- `tests/test_clip_zero_shot_classifier.py`: class ordering, no-training semantics, inference, errors, and persistence.
- `tests/test_entry_point.py`: installed metadata and component discovery.
- `tests/test_real_clip.py`: opt-in real-checkpoint smoke test.
- `README.md`: installation, GUI use, semantics, configuration, and testing.
- `LICENSE`: MIT license text.

---

### Task 1: Package Skeleton and Plugin Entry Point

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/dashai_clip_model_package/__init__.py`
- Create: `src/dashai_clip_model_package/clip_zero_shot_classifier.py`
- Create: `tests/test_entry_point.py`

**Interfaces:**
- Consumes: dashAI `BaseModel`, `BaseSchema`, and the `dashai.plugins` entry-point contract.
- Produces: importable `dashai_clip_model_package.CLIPZeroShotClassifier` and installed entry point named `CLIPZeroShotClassifier`.

- [ ] **Step 1: Write the failing public-import and metadata tests**

```python
from importlib.metadata import entry_points


def test_public_component_is_exported():
    from dashai_clip_model_package import CLIPZeroShotClassifier

    assert CLIPZeroShotClassifier.__name__ == "CLIPZeroShotClassifier"


def test_dashai_entry_point_loads_component():
    matches = [
        point
        for point in entry_points(group="dashai.plugins")
        if point.name == "CLIPZeroShotClassifier"
    ]
    assert len(matches) == 1
    assert matches[0].load().__name__ == "CLIPZeroShotClassifier"
```

- [ ] **Step 2: Run the import test and verify the package is absent**

Run: `python -m pytest tests/test_entry_point.py::test_public_component_is_exported -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'dashai_clip_model_package'`.

- [ ] **Step 3: Add package metadata and the minimal component**

Create `pyproject.toml` with these essential sections:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "dashai-clip-model-package"
version = "0.1.0"
description = "CLIP zero-shot image classification for dashAI"
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
authors = [{ name = "Diego Olguín-Wende", email = "dolguin@dim.uchile.cl" }]
keywords = ["DashAI", "Model"]
dependencies = [
  "dashAI>=0.9.7.post2",
  "numpy",
  "Pillow",
  "torch",
  "transformers",
]

[project.entry-points.'dashai.plugins']
CLIPZeroShotClassifier = "dashai_clip_model_package.clip_zero_shot_classifier:CLIPZeroShotClassifier"

[project.optional-dependencies]
dev = ["build", "pytest", "ruff"]

[tool.hatch.build.targets.wheel]
packages = ["src/dashai_clip_model_package"]

[tool.pytest.ini_options]
addopts = "-m 'not integration'"
markers = ["integration: downloads and executes a real Hugging Face checkpoint"]

[tool.ruff]
line-length = 88
target-version = "py310"
```

Create an initial `README.md` so editable installation and package builds can
resolve the declared readme before Task 7 expands it:

```markdown
# dashAI CLIP Model Package

CLIP zero-shot image classification for dashAI.
```

Create the minimal class:

```python
from DashAI.back.models.base_model import BaseModel


class CLIPZeroShotClassifier(BaseModel):
    COMPATIBLE_COMPONENTS = ["ImageClassificationTask"]
```

Export it from `__init__.py`:

```python
from dashai_clip_model_package.clip_zero_shot_classifier import (
    CLIPZeroShotClassifier,
)

__all__ = ["CLIPZeroShotClassifier"]
```

- [ ] **Step 4: Install editable and run the metadata tests**

Run: `python -m pip install -e '.[dev]'`

Run: `python -m pytest tests/test_entry_point.py -v`

Expected: both tests PASS. Abstract-method errors are addressed in Task 4 before instantiation tests.

- [ ] **Step 5: Commit the packaging checkpoint if Git is available**

```bash
git add pyproject.toml README.md src/dashai_clip_model_package tests/test_entry_point.py
git commit -m "build: scaffold dashAI CLIP plugin"
```

---

### Task 2: Prompt and Device Helpers

**Files:**
- Create: `src/dashai_clip_model_package/prompts.py`
- Create: `src/dashai_clip_model_package/device.py`
- Create: `tests/test_prompts.py`
- Create: `tests/test_device.py`

**Interfaces:**
- Produces: `validate_prompt_template(template: str) -> None`.
- Produces: `build_prompts(labels: Sequence[object], template: str) -> list[str]`.
- Produces: `resolve_device(requested: str) -> torch.device`.

- [ ] **Step 1: Write failing prompt tests**

```python
import pytest

from dashai_clip_model_package.prompts import build_prompts, validate_prompt_template


def test_build_prompts_normalizes_underscores():
    assert build_prompts(
        ["cat", "golden_retriever"], "a photo of a {}"
    ) == ["a photo of a cat", "a photo of a golden retriever"]


@pytest.mark.parametrize(
    "template",
    ["a photo", "{label}", "{} {}", "{!r}", "{:>10}"],
)
def test_validate_prompt_template_rejects_unsupported_fields(template):
    with pytest.raises(ValueError, match="exactly one plain"):
        validate_prompt_template(template)
```

- [ ] **Step 2: Run prompt tests and verify missing-module failure**

Run: `python -m pytest tests/test_prompts.py -v`

Expected: FAIL while importing `dashai_clip_model_package.prompts`.

- [ ] **Step 3: Implement deterministic prompt handling**

```python
from string import Formatter
from typing import Sequence


def validate_prompt_template(template: str) -> None:
    if not isinstance(template, str):
        raise TypeError("prompt_template must be a string")
    fields = [part for part in Formatter().parse(template) if part[1] is not None]
    valid = (
        len(fields) == 1
        and fields[0][1] == ""
        and fields[0][2] == ""
        and fields[0][3] is None
    )
    if not valid:
        raise ValueError(
            "prompt_template must contain exactly one plain positional '{}' field"
        )


def build_prompts(labels: Sequence[object], template: str) -> list[str]:
    validate_prompt_template(template)
    return [template.format(str(label).replace("_", " ")) for label in labels]
```

- [ ] **Step 4: Write failing device tests**

```python
import pytest
import torch

from dashai_clip_model_package.device import resolve_device


def test_auto_uses_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")


def test_cpu_is_explicit():
    assert resolve_device("cpu") == torch.device("cpu")


def test_cuda_unavailable_is_clear(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_device("cuda")


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="auto, cpu, or cuda"):
        resolve_device("mps")
```

- [ ] **Step 5: Implement device resolution and run both suites**

```python
import torch


def resolve_device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, or cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use device='auto' or 'cpu'")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Run: `python -m pytest tests/test_prompts.py tests/test_device.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the pure-helper checkpoint if Git is available**

```bash
git add src/dashai_clip_model_package/prompts.py src/dashai_clip_model_package/device.py tests/test_prompts.py tests/test_device.py
git commit -m "feat: add CLIP prompt and device helpers"
```

---

### Task 3: Shared Test Fakes and dashAI Schema

**Files:**
- Create: `tests/conftest.py`
- Modify: `src/dashai_clip_model_package/clip_zero_shot_classifier.py`
- Create: `tests/test_clip_zero_shot_classifier.py`

**Interfaces:**
- Consumes: `resolve_device()` and `validate_prompt_template()` from Task 2.
- Produces: `CLIPZeroShotClassifierSchema` and a constructible, lazy `CLIPZeroShotClassifier`.

- [ ] **Step 1: Add lightweight dataset and image fakes**

```python
from types import SimpleNamespace

import pytest
from PIL import Image


class FakeImage:
    def __init__(self, color):
        self.color = color

    def to_pil(self):
        return Image.new("RGB", (2, 2), self.color)


class FakeDataset:
    def __init__(self, column, values, column_type=None):
        self.column_names = [column]
        self.features = {column: object()}
        self.types = {column: column_type} if column_type is not None else {}
        self._column = column
        self._values = list(values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, key):
        if isinstance(key, int):
            return {self._column: self._values[key]}
        if key == self._column:
            return list(self._values)
        raise KeyError(key)


@pytest.fixture
def categorical_type():
    return SimpleNamespace(categories=["dog", "cat"])
```

- [ ] **Step 2: Write failing construction/schema test**

```python
from dashai_clip_model_package.clip_zero_shot_classifier import (
    CLIPZeroShotClassifier,
    CLIPZeroShotClassifierSchema,
)


def test_constructor_is_lazy_and_declares_compatible_task():
    component = CLIPZeroShotClassifier(device="cpu")
    assert component.model is None
    assert component.processor is None
    assert component.device.type == "cpu"
    assert component.COMPATIBLE_COMPONENTS == ["ImageClassificationTask"]
    assert component.SCHEMA is CLIPZeroShotClassifierSchema
```

- [ ] **Step 3: Run test and verify abstract/schema failure**

Run: `python -m pytest tests/test_clip_zero_shot_classifier.py::test_constructor_is_lazy_and_declares_compatible_task -v`

Expected: FAIL because the minimal class lacks schema and abstract method implementations.

- [ ] **Step 4: Add schema, metadata, constructor, and method shells with explicit errors**

Implement schema fields with `string_field`, `int_field(ge=1)`, and
`enum_field(enum=["auto", "cpu", "cuda"])`; each `schema_field` gets a
placeholder, `MultilingualString` alias, and description. Add:

```python
class CLIPZeroShotClassifier(BaseModel):
    SCHEMA = CLIPZeroShotClassifierSchema
    COMPATIBLE_COMPONENTS = ["ImageClassificationTask"]
    DISPLAY_NAME = MultilingualString(en="CLIP Zero-Shot", es="CLIP Zero-Shot")
    DESCRIPTION = MultilingualString(
        en="Classify images by comparing them with text prompts, without fine-tuning.",
        es="Clasifica imágenes comparándolas con prompts de texto, sin ajuste fino.",
    )
    COLOR = "#5B4BDB"
    ICON = "ImageSearch"

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        prompt_template="a photo of a {}",
        batch_size=32,
        device="auto",
        **kwargs,
    ):
        validate_prompt_template(prompt_template)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        self.device_name = device
        self.device = resolve_device(device)
        self.model = None
        self.processor = None
        self.class_names = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        self._text_features = None

    def train(self, x_train, y_train, x_validation=None, y_validation=None):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def save(self, filename):
        raise NotImplementedError

    @classmethod
    def load(cls, filename):
        raise NotImplementedError
```

- [ ] **Step 5: Run the construction test**

Run: `python -m pytest tests/test_clip_zero_shot_classifier.py::test_constructor_is_lazy_and_declares_compatible_task -v`

Expected: PASS without Hugging Face downloads.

- [ ] **Step 6: Commit the schema/lifecycle checkpoint if Git is available**

```bash
git add src/dashai_clip_model_package/clip_zero_shot_classifier.py tests/conftest.py tests/test_clip_zero_shot_classifier.py
git commit -m "feat: define dashAI CLIP component schema"
```

---

### Task 4: Class Ordering and Zero-Training Lifecycle

**Files:**
- Modify: `src/dashai_clip_model_package/clip_zero_shot_classifier.py`
- Modify: `tests/test_clip_zero_shot_classifier.py`

**Interfaces:**
- Produces: `train(...) -> CLIPZeroShotClassifier` with canonical class metadata.
- Produces: `prepare_output(dataset, is_fit=False) -> DashAIDataset`.
- Produces: `_ensure_backend() -> None` and `_prepare_text_features() -> None` extension points used by Task 5 and tests.

- [ ] **Step 1: Write failing canonical-order and no-training tests**

```python
def test_train_preserves_categorical_order(monkeypatch, categorical_type):
    component = CLIPZeroShotClassifier(device="cpu")
    monkeypatch.setattr(component, "_ensure_backend", lambda: None)
    monkeypatch.setattr(component, "_prepare_text_features", lambda: None)
    x = FakeDataset("image", [FakeImage("red"), FakeImage("blue")])
    y = FakeDataset("label", ["cat", "dog"], categorical_type)

    result = component.train(x, y)

    assert result is component
    assert component.class_names == ["dog", "cat"]
    assert component.label_to_idx == {"dog": 0, "cat": 1}
    assert component.idx_to_label == {0: "dog", 1: "cat"}


def test_train_fallback_uses_first_seen_order(monkeypatch):
    component = CLIPZeroShotClassifier(device="cpu")
    monkeypatch.setattr(component, "_ensure_backend", lambda: None)
    monkeypatch.setattr(component, "_prepare_text_features", lambda: None)
    y = FakeDataset("label", ["bird", "cat", "bird", "dog"])

    component.train(FakeDataset("image", []), y)

    assert component.class_names == ["bird", "cat", "dog"]
```

- [ ] **Step 2: Run both tests and verify `NotImplementedError`**

Run: `python -m pytest tests/test_clip_zero_shot_classifier.py -k 'train_preserves or train_fallback' -v`

Expected: FAIL in `train()`.

- [ ] **Step 3: Implement ordered-label extraction and zero-training lifecycle**

```python
def _extract_class_names(self, y_train):
    if len(y_train.column_names) != 1:
        raise ValueError("CLIPZeroShotClassifier requires exactly one output column")
    column = y_train.column_names[0]
    output_type = (getattr(y_train, "types", {}) or {}).get(column)
    categories = getattr(output_type, "categories", None)
    if categories:
        names = list(categories)
    else:
        names = list(dict.fromkeys(y_train[column]))
    if not names:
        raise ValueError("At least one class label is required")
    return names


def train(self, x_train, y_train, x_validation=None, y_validation=None):
    self.class_names = self._extract_class_names(y_train)
    self.label_to_idx = {label: index for index, label in enumerate(self.class_names)}
    self.idx_to_label = {index: label for label, index in self.label_to_idx.items()}
    self._text_features = None
    self._ensure_backend()
    self._prepare_text_features()
    return self
```

Implement `_ensure_backend()` using lazy `CLIPModel.from_pretrained()` and
`CLIPProcessor.from_pretrained()` with errors wrapped as:

```python
raise RuntimeError(f"Unable to load CLIP checkpoint '{self.model_name}'") from exc
```

Implement `_prepare_text_features()` with `build_prompts()`, processor text
inputs, tensor movement, `torch.inference_mode()`, `get_text_features()`, and
L2 normalization using `clamp_min(torch.finfo(dtype).eps)`.

- [ ] **Step 4: Add and pass parameter-immutability test**

```python
import torch


def test_train_does_not_modify_model_parameters(monkeypatch):
    component = CLIPZeroShotClassifier(device="cpu")
    component.model = torch.nn.Linear(2, 2)
    before = [parameter.detach().clone() for parameter in component.model.parameters()]
    flags = [parameter.requires_grad for parameter in component.model.parameters()]
    monkeypatch.setattr(component, "_ensure_backend", lambda: None)
    monkeypatch.setattr(component, "_prepare_text_features", lambda: None)
    y = FakeDataset("label", ["cat", "dog"])

    component.train(FakeDataset("image", []), y)

    after = list(component.model.parameters())
    assert all(torch.equal(old, new) for old, new in zip(before, after))
    assert [parameter.requires_grad for parameter in after] == flags
```

Run: `python -m pytest tests/test_clip_zero_shot_classifier.py -k train -v`

Expected: PASS.

- [ ] **Step 5: Add `prepare_output()` using dashAI's dataset type**

```python
def prepare_output(self, dataset, is_fit=False):
    import pyarrow as pa
    from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset

    if not self.label_to_idx:
        return dataset
    column = dataset.column_names[0]
    try:
        encoded = [self.label_to_idx[value] for value in dataset[column]]
    except KeyError as exc:
        raise ValueError(f"Unknown class label: {exc.args[0]!r}") from exc
    return DashAIDataset(pa.table({column: encoded}))
```

Test that `['cat', 'dog']` becomes `[1, 0]` for canonical order
`['dog', 'cat']`.

- [ ] **Step 6: Commit the zero-shot training checkpoint if Git is available**

```bash
git add src/dashai_clip_model_package/clip_zero_shot_classifier.py tests/test_clip_zero_shot_classifier.py
git commit -m "feat: capture zero-shot CLIP class metadata"
```

---

### Task 5: Batched CLIP Probability Inference

**Files:**
- Modify: `tests/conftest.py`
- Modify: `src/dashai_clip_model_package/clip_zero_shot_classifier.py`
- Modify: `tests/test_clip_zero_shot_classifier.py`

**Interfaces:**
- Consumes: ordered `class_names`, cached normalized `_text_features`, resolved `device`.
- Produces: `predict(x) -> numpy.ndarray` with shape `(N, K)` and `float32` dtype.

- [ ] **Step 1: Add deterministic processor/model fakes**

```python
from types import SimpleNamespace

import torch


class FakeProcessor:
    def __init__(self):
        self.image_batch_sizes = []

    def __call__(self, *, images=None, text=None, return_tensors="pt", padding=False):
        if images is not None:
            self.image_batch_sizes.append(len(images))
            rows = []
            for image in images:
                red, _green, blue = image.getpixel((0, 0))
                rows.append([float(red > blue), float(blue > red)])
            return {"pixel_values": torch.tensor(rows)}
        return {"input_ids": torch.arange(len(text)).reshape(-1, 1)}


class FakeCLIPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))

    def get_image_features(self, pixel_values):
        return pixel_values

    def get_text_features(self, input_ids):
        return torch.eye(len(input_ids), 2)


@pytest.fixture
def fake_backend():
    return SimpleNamespace(
        model=FakeCLIPModel(),
        processor=FakeProcessor(),
        text_features=torch.eye(2),
    )
```

This makes expected class order and batch order observable without network
access.

- [ ] **Step 2: Write the failing probability and batching test**

```python
import numpy as np


def test_predict_returns_ordered_probabilities_in_batches(fake_backend):
    component = CLIPZeroShotClassifier(batch_size=2, device="cpu")
    component.model = fake_backend.model
    component.processor = fake_backend.processor
    component.class_names = ["red", "blue"]
    component.label_to_idx = {"red": 0, "blue": 1}
    component.idx_to_label = {0: "red", 1: "blue"}
    component._text_features = fake_backend.text_features
    dataset = FakeDataset(
        "image",
        [FakeImage("red"), FakeImage("blue"), FakeImage("red")],
    )

    probabilities = component.predict(dataset)

    assert probabilities.shape == (3, 2)
    assert probabilities.dtype == np.float32
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(probabilities >= 0)
    assert probabilities.argmax(axis=1).tolist() == [0, 1, 0]
    assert component.processor.image_batch_sizes == [2, 1]
```

- [ ] **Step 3: Run the test and verify `predict()` is unimplemented**

Run: `python -m pytest tests/test_clip_zero_shot_classifier.py::test_predict_returns_ordered_probabilities_in_batches -v`

Expected: FAIL in `predict()`.

- [ ] **Step 4: Implement image conversion and batched inference**

Add `_to_pil_image(value, index)` that requires `to_pil()`, converts to RGB,
and raises `ValueError(f"Unable to decode image at sample {index}")` from the
original exception.

Implement `predict()` to validate fitted state and one image column, return
`np.empty((0, len(self.class_names)), dtype=np.float32)` for no samples, and
loop over `range(0, len(x), self.batch_size)`. For every batch:

```python
image_inputs = self.processor(images=images, return_tensors="pt")
image_inputs = {name: value.to(self.device) for name, value in image_inputs.items()}
with torch.inference_mode():
    image_features = self.model.get_image_features(**image_inputs)
    denominator = image_features.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(image_features.dtype).eps
    )
    image_features = image_features / denominator
    logits = self.model.logit_scale.exp() * image_features @ self._text_features.T
    batches.append(torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32))
return np.concatenate(batches, axis=0)
```

Call `_ensure_backend()`, `_prepare_text_features()` when cache is absent,
`model.to(self.device)`, and `model.eval()` before the loop.

- [ ] **Step 5: Add focused error and edge-case tests**

Add tests proving:

- prediction before class setup raises `RuntimeError` containing `train or load`;
- empty input returns `(0, K)`;
- a non-image value reports its sample index;
- more than one input column raises an exact-one-column error;
- a simulated `torch.cuda.OutOfMemoryError` is re-raised with a batch-size hint
  and the original exception as `__cause__`.

Run: `python -m pytest tests/test_clip_zero_shot_classifier.py -v`

Expected: PASS.

- [ ] **Step 6: Commit inference if Git is available**

```bash
git add src/dashai_clip_model_package/clip_zero_shot_classifier.py tests/conftest.py tests/test_clip_zero_shot_classifier.py
git commit -m "feat: add batched CLIP zero-shot inference"
```

---

### Task 6: Lightweight Persistence

**Files:**
- Modify: `src/dashai_clip_model_package/clip_zero_shot_classifier.py`
- Modify: `tests/test_clip_zero_shot_classifier.py`

**Interfaces:**
- Produces: `save(filename: str | Path) -> None`.
- Produces: `load(filename: str | Path) -> CLIPZeroShotClassifier`.
- Checkpoint schema version: integer `1`.

- [ ] **Step 1: Write the failing round-trip test**

```python
def test_save_load_round_trip_is_lazy(tmp_path):
    component = CLIPZeroShotClassifier(
        model_name="org/checkpoint",
        prompt_template="an image of {}",
        batch_size=7,
        device="cpu",
    )
    component.class_names = ["dog", "cat"]
    component.label_to_idx = {"dog": 0, "cat": 1}
    component.idx_to_label = {0: "dog", 1: "cat"}
    component.model = object()
    component.processor = object()
    path = tmp_path / "clip.pt"

    component.save(path)
    restored = CLIPZeroShotClassifier.load(path)

    assert restored.model_name == "org/checkpoint"
    assert restored.prompt_template == "an image of {}"
    assert restored.batch_size == 7
    assert restored.class_names == ["dog", "cat"]
    assert restored.model is None
    assert restored.processor is None
    assert restored._text_features is None
```

- [ ] **Step 2: Run and verify persistence is unimplemented**

Run: `python -m pytest tests/test_clip_zero_shot_classifier.py::test_save_load_round_trip_is_lazy -v`

Expected: FAIL in `save()`.

- [ ] **Step 3: Implement the versioned state-only checkpoint**

```python
def save(self, filename):
    import torch

    torch.save(
        {
            "format_version": 1,
            "model_name": self.model_name,
            "prompt_template": self.prompt_template,
            "batch_size": self.batch_size,
            "device_name": self.device_name,
            "class_names": self.class_names,
        },
        filename,
    )


@classmethod
def load(cls, filename):
    import torch

    state = torch.load(filename, map_location="cpu", weights_only=True)
    required = {
        "format_version", "model_name", "prompt_template", "batch_size",
        "device_name", "class_names",
    }
    if not isinstance(state, dict) or required - state.keys():
        raise ValueError("Invalid CLIPZeroShotClassifier checkpoint")
    if state["format_version"] != 1:
        raise ValueError("Unsupported CLIPZeroShotClassifier checkpoint version")
    instance = cls(
        model_name=state["model_name"],
        prompt_template=state["prompt_template"],
        batch_size=state["batch_size"],
        device=state["device_name"],
    )
    instance.class_names = list(state["class_names"])
    instance.label_to_idx = {
        label: index for index, label in enumerate(instance.class_names)
    }
    instance.idx_to_label = {
        index: label for label, index in instance.label_to_idx.items()
    }
    return instance
```

- [ ] **Step 4: Add malformed/version tests and run the suite**

Test missing keys and `format_version=2`, each with its documented error.

Run: `python -m pytest tests/test_clip_zero_shot_classifier.py -v`

Expected: PASS.

- [ ] **Step 5: Commit persistence if Git is available**

```bash
git add src/dashai_clip_model_package/clip_zero_shot_classifier.py tests/test_clip_zero_shot_classifier.py
git commit -m "feat: persist CLIP configuration and labels"
```

---

### Task 7: Real-Model Smoke Test and User Documentation

**Files:**
- Create: `tests/test_real_clip.py`
- Modify: `README.md`
- Create: `LICENSE`

**Interfaces:**
- Consumes: the complete public component API.
- Produces: opt-in integration coverage and complete installation/usage guidance.

- [ ] **Step 1: Add the opt-in real-model smoke test**

```python
import os

import pytest

from dashai_clip_model_package import CLIPZeroShotClassifier


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_CLIP_INTEGRATION") != "1",
    reason="set RUN_CLIP_INTEGRATION=1 to download and run CLIP",
)
def test_real_clip_smoke(fake_two_class_dashai_dataset):
    x, y = fake_two_class_dashai_dataset
    component = CLIPZeroShotClassifier(batch_size=2, device="cpu")
    component.train(x, y)
    probabilities = component.predict(x)
    assert probabilities.shape == (2, 2)
```

Define the fixture in `tests/conftest.py` using the already declared fakes:

```python
@pytest.fixture
def fake_two_class_dashai_dataset():
    categorical = SimpleNamespace(categories=["red", "blue"])
    x = FakeDataset("image", [FakeImage("red"), FakeImage("blue")])
    y = FakeDataset("label", ["red", "blue"], categorical)
    return x, y
```

Run default suite: `python -m pytest -v`

Expected: integration test is deselected by marker expression and all unit tests PASS.

- [ ] **Step 2: Write README with executable commands and semantic warning**

Document:

- `pip install dashai-clip-model-package` for published use;
- `pip install -e '.[dev]'` for local development;
- GUI steps: dataset, `ImageClassificationTask`, component selection,
  configuration, run/evaluate;
- parameter table with defaults and device behavior;
- `cat`, `dog`, `car` prompt example;
- bold statement that `train()` does not fine-tune CLIP;
- lightweight persistence and checkpoint re-download/cache requirement;
- `python -m pytest`, Ruff, build, and
  `RUN_CLIP_INTEGRATION=1 python -m pytest -m integration -v` commands.

- [ ] **Step 3: Add the standard MIT license**

Use copyright line:

```text
Copyright (c) 2026 Diego Olguín-Wende
```

- [ ] **Step 4: Run documentation-facing package checks**

Run: `python -m build`

Run: `python -m pip install --force-reinstall --no-deps dist/dashai_clip_model_package-0.1.0-py3-none-any.whl`

Run: `python -m pytest tests/test_entry_point.py -v`

Expected: wheel and sdist build; installed entry point loads successfully.

- [ ] **Step 5: Commit docs/integration test if Git is available**

```bash
git add README.md LICENSE tests/test_real_clip.py
git commit -m "docs: document dashAI CLIP plugin usage"
```

---

### Task 8: Full Verification and Acceptance Audit

**Files:**
- Modify only files implicated by verification failures.

**Interfaces:**
- Consumes: all prior task deliverables.
- Produces: an installable, lint-clean, test-passing plugin satisfying every acceptance criterion.

- [ ] **Step 1: Run formatting and lint checks without mutation**

Run: `python -m ruff format --check .`

Run: `python -m ruff check .`

Expected: both exit 0. If either fails, make only the reported changes and rerun both commands.

- [ ] **Step 2: Run all default tests**

Run: `python -m pytest -v`

Expected: all non-integration tests PASS; no Hugging Face download begins.

- [ ] **Step 3: Build and inspect artifacts**

Run: `python -m build`

Run: `python -m zipfile -l dist/dashai_clip_model_package-0.1.0-py3-none-any.whl`

Expected: the wheel contains the three package modules and distribution metadata, with no tests, cached weights, or development artifacts.

- [ ] **Step 4: Verify the installed entry point in a clean virtual environment**

Create a temporary environment, install the already built wheel with the
current dashAI-compatible dependency set, and execute:

```python
from importlib.metadata import entry_points

point = next(
    point
    for point in entry_points(group="dashai.plugins")
    if point.name == "CLIPZeroShotClassifier"
)
component = point.load()
assert component.COMPATIBLE_COMPONENTS == ["ImageClassificationTask"]
```

Expected: process exits 0 without attempting to download CLIP weights.

- [ ] **Step 5: Audit the specification checklist**

Confirm from test output and package inspection that discovery, schema,
canonical labels, CPU/auto/CUDA behavior, dashAI image conversion, `(N, K)`
probabilities, metric-compatible label encoding, persistence, documentation,
and no core changes are all covered. Record a real-model test as optional if
network or model download is unavailable; do not claim it passed unless run.

- [ ] **Step 6: Commit final verification fixes if Git is available**

```bash
git add pyproject.toml src tests README.md LICENSE docs
git commit -m "test: verify dashAI CLIP plugin release"
```
