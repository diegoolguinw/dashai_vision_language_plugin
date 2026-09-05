# Codex Implementation Spec — CLIP Plugin for dashAI

## 0. Purpose

Implement a **dashAI plugin that adds CLIP-based zero-shot image classification**, using Hugging Face Transformers as the backend.

The first contribution should be intentionally small, compatible with dashAI's existing image-classification pipeline, and suitable for an upstream/open-source contribution.

Do **not** redesign dashAI's core multimodal architecture in this first implementation.

A later phase may add CLIP embeddings, image–text retrieval, and general vision-language abstractions, but those are explicitly out of scope for the MVP unless they are required by the current dashAI interfaces.

---

# 1. High-level objective

Create a Python package/plugin, tentatively named:

```text
dashai-clip-model-package
```

that allows a dashAI user to select a CLIP model and run **zero-shot image classification** on a standard dashAI `ImageClassificationTask` dataset.

The plugin should:

1. Load a pretrained CLIP checkpoint.
2. Obtain the class labels from the dashAI image-classification dataset/task.
3. Convert labels into text prompts.
4. Encode images and text labels with CLIP.
5. Compute image–text cosine similarities using CLIP's learned logit scale when appropriate.
6. Return class probabilities in the format expected by dashAI's image-classification pipeline.
7. Work on CPU and CUDA when available.
8. Be automatically discoverable by dashAI through its plugin entry-point system.
9. Include unit tests and basic integration tests.
10. Include clear installation and usage documentation.

---

# 2. Repository and version assumptions

Use the **current dashAI repository**, not assumptions from older versions.

Repository:

```text
https://github.com/DashAISoftware/dashAI
```

Before implementing anything:

1. Clone/update the repository.
2. Inspect the current default/development branches.
3. Read the root `CLAUDE.md`, `CONTRIBUTING.rst`, and plugin-development documentation if present.
4. Inspect the current plugin system and component registry.
5. Inspect the official image-classification plugin/package.
6. Identify the exact image-classification model base class used by the current version.
7. Identify the exact required signatures for training, prediction, saving/loading, schema declaration, and metadata.
8. Do not invent APIs that are not present in the checked-out code.

Useful references:

```text
https://docs.dash-ai.com/build/plugin-development/develop/
https://docs.dash-ai.com/build/plugin-development/structure/
https://docs.dash-ai.com/components/tasks/ImageClassificationTask/
https://github.com/DashAISoftware/dashai-image-classification-package
```

Current dashAI architecture uses a component registry and Python `entry_points` under the group:

```text
dashai.plugins
```

Confirm the exact current mechanism from the repository before writing `pyproject.toml`.

---

# 3. Design principle

## Reuse the existing image-classification abstraction

CLIP zero-shot classification should behave like a regular image-classification model from dashAI's point of view.

Conceptually:

```text
ImageClassificationTask
          |
          v
CLIPZeroShotClassifier
          |
          v
class probability matrix
          |
          v
Accuracy / F1 / LogLoss / etc.
```

The plugin should not require a new task type for the MVP.

The central abstraction is:

Given class labels

```text
[class_1, ..., class_K]
```

construct prompts

```text
[a photo of a class_1, ..., a photo of a class_K]
```

or according to a configurable prompt template.

For an image `x`, compute normalized embeddings

```math
z_I = f_\theta(x) / ||f_\theta(x)||,
```

and, for prompt `t_j`,

```math
z_{T,j} = g_\phi(t_j) / ||g_\phi(t_j)||.
```

Then compute logits

```math
\ell_j(x) = s\, z_I^\top z_{T,j},
```

where `s` is CLIP's learned logit scale when exposed by the selected backend.

Return

```math
p_j(x) = \operatorname{softmax}(\ell(x))_j.
```

The returned matrix should have shape

```text
(n_samples, n_classes)
```

in the class ordering expected by dashAI.

---

# 4. Backend

Use Hugging Face Transformers for the MVP.

Preferred imports:

```python
from transformers import CLIPModel, CLIPProcessor
```

Default checkpoint:

```text
openai/clip-vit-base-patch32
```

The checkpoint should be configurable.

At minimum support arbitrary Hugging Face model identifiers compatible with `CLIPModel` / `CLIPProcessor`.

Do not hard-code a list of only three OpenAI models unless dashAI's schema requires a finite choice list. If free strings are allowed, prefer a string parameter with the above default.

Dependencies should be kept minimal. Likely requirements:

```text
transformers
torch
Pillow
```

but inspect what dashAI itself already depends on before duplicating constraints.

---

# 5. Proposed public component

Tentative class name:

```python
CLIPZeroShotClassifier
```

If dashAI has a naming convention such as `CLIPZeroShotImageClassificationModel`, follow the repository convention instead.

The component should extend the **current dashAI image-classification model base class**, not plain `BaseModel`, unless the current architecture indicates otherwise.

Before implementation, inspect classes analogous to:

```text
ResNet
ViT
image classification plugin models
```

and mirror their contracts closely.

---

# 6. Configuration parameters

Expose only parameters that are useful to a GUI user.

Recommended MVP parameters:

## `model_name`

```text
str
```

Default:

```text
openai/clip-vit-base-patch32
```

Description: Hugging Face identifier of a CLIP-compatible checkpoint.

## `prompt_template`

```text
str
```

Default:

```text
a photo of a {}
```

The template must contain a formatting placeholder for the class label.

Accept either `{}` or, if easier to validate, `{label}`. Pick one convention and document it.

Examples:

```text
a photo of a {}
a photo of the {}
an image of a {}
{}
```

## `batch_size`

```text
int
```

Suggested default:

```text
32
```

Use a reasonable range according to dashAI's schema system.

## `device`

Preferred values:

```text
auto
cpu
cuda
```

Default:

```text
auto
```

Behavior:

```text
auto -> CUDA if torch.cuda.is_available(), otherwise CPU
cuda -> fail with a clear error if CUDA is unavailable
cpu  -> CPU
```

Do not add many low-level inference options in the first version.

Optional only if easy and idiomatic in dashAI:

```text
normalize_class_names: bool
```

but it is not necessary for the MVP.

---

# 7. Training semantics

CLIP zero-shot classification does not train model parameters.

However, dashAI's model abstraction may require a `fit`, `train`, or equivalent method.

Implement that method according to the current contract while preserving the correct semantics:

- Do not fine-tune CLIP.
- Extract and store the ordered class labels required by the task.
- Prepare/cache text embeddings for those labels if appropriate.
- Mark the component as fitted/ready if the framework expects that state.

If the dashAI API receives `X` and `y` rather than task metadata, derive the unique class ordering in exactly the way expected by the existing classification base classes.

Do not independently sort labels if dashAI supplies an explicit canonical ordering.

The class ordering used to build prompts must be identical to the ordering of the output probability columns.

---

# 8. Prompt generation

For labels such as:

```text
cat
golden_retriever
fire_truck
```

generate human-readable prompts where reasonable.

A minimal safe normalization is:

```python
label.replace("_", " ")
```

Do not perform aggressive language rewriting.

Example:

```text
golden_retriever -> golden retriever
```

Prompt:

```text
a photo of a golden retriever
```

The transformation should be deterministic and unit tested.

---

# 9. Prediction path

Implement batched inference.

Pseudocode only — adapt to dashAI's actual dataset API:

```python
prompts = [prompt_template.format(label) for label in class_names]

text_inputs = processor(
    text=prompts,
    return_tensors="pt",
    padding=True,
)

with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

for batch in images:
    image_inputs = processor(images=batch, return_tensors="pt")

    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.T
        probabilities = logits.softmax(dim=-1)
```

Important:

- Determine from the Transformers version whether `model.logit_scale.exp()` or another API is the correct learned temperature scaling.
- Avoid gradients (`torch.inference_mode()` is preferable when supported).
- Move tensors to the selected device.
- Return NumPy arrays if dashAI expects NumPy.
- The output must contain probabilities, not only argmax labels, if existing classification metrics require probabilities.

---

# 10. Dataset/image handling

Inspect the current `ImageClassificationTask.prepare_for_task(...)` output and the image-classification model implementations.

Support exactly the image representation used by dashAI, e.g. one of:

```text
PIL.Image
Hugging Face datasets Image feature
filesystem path
NumPy array
```

Do not assume the representation before inspecting the actual implementation.

Create a small adapter function if needed, for example:

```python
def _to_pil_image(value) -> Image.Image: ...
```

but only if the current framework requires it.

The plugin must work with datasets loaded through dashAI's normal UI image-classification workflow.

---

# 11. Persistence

Inspect how current dashAI models are saved and loaded.

Desired behavior:

- Do not serialize the entire Hugging Face pretrained checkpoint into dashAI run state unless existing model plugins do so.
- Persist configuration and learned runtime state needed to reconstruct the model.
- At minimum persist:
  - `model_name`
  - prompt template
  - class names / label ordering
  - any dashAI-required metadata

On load, reconstruct the Hugging Face model from `model_name`.

If dashAI's framework automatically pickles model instances, ensure CLIP objects do not create an unnecessary or broken serialized artifact. Follow the pattern used by other Hugging Face or diffusion plugins.

---

# 12. Plugin package structure

Use the current dashAI plugin template/convention.

Tentative structure:

```text
dashai-clip-model-package/
├── LICENSE
├── README.md
├── pyproject.toml
├── src/
│   └── dashai_clip_model_package/
│       ├── __init__.py
│       ├── clip_zero_shot_classifier.py
│       └── ... schema/support files if required
└── tests/
    ├── test_clip_zero_shot_classifier.py
    └── ...
```

If official dashAI plugins do not use a `src/` layout, follow the current first-party plugin convention instead.

Consistency with current dashAI plugins is more important than this tentative tree.

---

# 13. `pyproject.toml`

Mirror the current official dashAI plugins.

Requirements:

- package name starts with `dashai-`
- declare Python version compatible with current dashAI
- declare runtime dependencies
- register the plugin through the current `dashai.plugins` entry-point mechanism

Do not guess the entry-point value syntax. Copy/adapt it from a current first-party plugin.

---

# 14. dashAI schema / GUI metadata

Current dashAI components expose parameter schemas that drive frontend forms.

Inspect the current mechanism (`BaseSchema`, schema fields, JSON schema files, or whichever mechanism the current plugin API uses).

Create GUI-visible configuration for:

```text
Model name
Prompt template
Batch size
Device
```

Descriptions should be understandable by non-expert users.

Example wording:

```text
Model name:
Hugging Face CLIP checkpoint used for image and text encoding.

Prompt template:
Template used to convert each class label into a natural-language CLIP prompt.

Batch size:
Number of images processed per inference batch.

Device:
Hardware used for inference. 'auto' selects CUDA when available.
```

Follow dashAI's multilingual-string conventions if required.

---

# 15. Tests

Tests must not depend on downloading a full CLIP model on every regular CI run if avoidable.

Use mocking/unit separation where appropriate.

## Unit tests

At minimum test:

### A. Prompt construction

Input:

```text
["cat", "golden_retriever"]
```

Expected prompts:

```text
["a photo of a cat", "a photo of a golden retriever"]
```

### B. Device resolution

Test:

```text
auto + CUDA unavailable -> cpu
cpu -> cpu
cuda + CUDA unavailable -> informative exception
```

### C. Probability output

With mocked image/text embeddings:

- output shape is `(N, K)`
- each row sums approximately to 1
- probabilities are non-negative
- class ordering is preserved

### D. Batching

Ensure `N > batch_size` gives the same ordered number of predictions.

### E. No-training behavior

Ensure the training/fit step does not modify pretrained model parameters and correctly stores class labels.

## Integration test

Provide one optional/slower integration test, marked appropriately, using a very small dataset and a real CLIP checkpoint.

If Hugging Face provides a tiny random CLIP test checkpoint compatible with the installed Transformers version, prefer that for CI.

Otherwise mock the HF backend for default CI and document how to run a real-model smoke test manually.

---

# 16. Code quality

Follow the repository's current tooling and conventions.

Before declaring completion, run all relevant commands, such as whichever apply:

```bash
ruff check .
ruff format --check .
pytest
mypy ...
pre-commit run --all-files
```

Do not blindly run commands that are not configured. Inspect `pyproject.toml` and repository CI first.

Add type annotations.

Add docstrings consistent with dashAI's style.

Do not leave commented-out experiments or debug print statements.

---

# 17. README requirements

The plugin README should explain:

## What it does

CLIP zero-shot image classification inside dashAI.

## Installation

For development, document a local editable installation consistent with dashAI's plugin workflow.

Example, only if appropriate after inspecting current setup:

```bash
pip install -e ./plugins/dashai-clip-model-package
```

Also document normal package installation if/when published:

```bash
pip install dashai-clip-model-package
```

## Usage

Explain the GUI workflow:

```text
1. Load an image-classification dataset.
2. Choose ImageClassificationTask.
3. Select CLIPZeroShotClassifier.
4. Configure model/prompt/device.
5. Run prediction/evaluation.
```

## Important semantic note

Explain clearly that this is **zero-shot inference**, not supervised fine-tuning.

## Example

For classes:

```text
cat
dog
car
```

and template:

```text
a photo of a {}
```

CLIP compares each image to the text representations of those three prompts.

---

# 18. Acceptance criteria — MVP

The implementation is complete only when all of the following are true:

- [ ] The package installs successfully in a clean environment compatible with dashAI.
- [ ] dashAI discovers the plugin automatically.
- [ ] The CLIP model appears as compatible with `ImageClassificationTask`.
- [ ] A user can choose a Hugging Face CLIP checkpoint.
- [ ] A user can configure the prompt template.
- [ ] CPU inference works.
- [ ] `device="auto"` behaves correctly.
- [ ] CUDA inference is supported when PyTorch/CUDA is available.
- [ ] The model receives dashAI image datasets without requiring manual conversion outside the UI.
- [ ] The model obtains/stores the correct ordered class names.
- [ ] Prediction returns an `(n_samples, n_classes)` probability matrix in the correct class order.
- [ ] Existing dashAI image-classification metrics can consume the output.
- [ ] Unit tests pass.
- [ ] A real-model smoke test is documented or implemented.
- [ ] README is complete.
- [ ] No changes to dashAI core are required unless a genuine incompatibility is discovered.

---

# 19. Explicit non-goals for the first PR

Do **not** implement the following in the MVP:

- CLIP fine-tuning.
- Contrastive CLIP training.
- Image–text retrieval task.
- Recall@K retrieval metrics.
- General `VisionLanguageModel` base class.
- UMAP/PCA embedding explorer.
- Modality-gap visualization.
- OpenCLIP backend.
- SigLIP.
- BLIP/BLIP-2.
- Prompt ensembling.
- Learnable prompts / CoOp.

These are natural follow-up contributions but should not complicate the first plugin.

---

# 20. Phase 2 roadmap — do not implement unless requested

After the MVP is working, propose a second contribution around multimodal representations.

Possible components:

## A. CLIP image embedding converter

```text
image -> R^d
```

## B. CLIP text embedding converter

```text
text -> R^d
```

## C. Image–text retrieval task

Dataset:

```math
\{(x_i,t_i)\}_{i=1}^N
```

Similarity matrix:

```math
S_{ij}=\frac{f(x_i)^\top g(t_j)}{\|f(x_i)\|\|g(t_j)\|}.
```

Metrics:

```text
Image -> Text Recall@1, Recall@5, Recall@10
Text -> Image Recall@1, Recall@5, Recall@10
Median Rank
Mean Rank
```

## D. General vision-language abstraction

Only after studying dashAI architecture and discussing with maintainers, consider something conceptually like:

```python
class VisionLanguageModel(...):
    def encode_image(self, images): ...
    def encode_text(self, texts): ...
    def similarity(self, images, texts): ...
```

This should be an upstream architectural discussion, not silently introduced as part of the zero-shot classifier plugin.

---

# 21. Suggested implementation workflow for Codex

Work incrementally.

## Step 1 — repository inspection

Report:

- current relevant branch
- dashAI version
- exact image-classification base class
- exact method signatures
- existing official image-classification models
- plugin entry-point syntax
- schema mechanism
- persistence mechanism
- test conventions

Do this before making architectural decisions.

## Step 2 — write an implementation plan

List files to create/modify and map each file to an acceptance criterion.

## Step 3 — implement tests first where practical

At minimum write tests for prompt construction, class ordering, output probabilities, and device selection before implementing the corresponding helpers.

## Step 4 — implement MVP

Keep changes inside the plugin package whenever possible.

## Step 5 — run verification

Run plugin tests and any relevant dashAI integration tests.

## Step 6 — manual smoke test

Launch dashAI with the local plugin installed and verify that the component appears and can run on a tiny image dataset.

## Step 7 — final report

Report:

- files added/changed
- architecture used
- exact commands run
- test results
- manual test result
- any dashAI limitations discovered
- follow-up recommendations

Do not claim that the plugin works until the verification commands have actually passed.

---

# 22. Handling incompatibilities

If the current dashAI `ImageClassificationModel` contract fundamentally assumes supervised training in a way that prevents zero-shot CLIP from being represented cleanly:

1. Do not force a brittle workaround.
2. Identify the smallest incompatibility precisely.
3. Search existing dashAI patterns for pretrained/non-trainable models.
4. Propose the minimum framework extension needed.
5. Keep the extension general rather than CLIP-specific.
6. Separate core changes from plugin changes in commits.

Before introducing a core API change, explain why it is necessary and what existing behavior it preserves.

---

# 23. Preferred Git history

Use small coherent commits, for example:

```text
test: add CLIP zero-shot classifier unit tests
feat: add CLIP zero-shot image classifier
feat: register CLIP dashAI plugin
 docs: add CLIP plugin usage guide
```

Adjust wording to repository conventions.

---

# 24. Deliverable

The final deliverable should be a working repository/package that can be proposed upstream as:

> Add CLIP zero-shot image classification support to dashAI through a standalone plugin compatible with the existing ImageClassificationTask.

The implementation should optimize for:

1. correctness,
2. compatibility with current dashAI architecture,
3. minimal core changes,
4. maintainability,
5. clear tests,
6. future extensibility toward multimodal image–text tasks.

