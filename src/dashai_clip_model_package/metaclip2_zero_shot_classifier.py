from typing import ClassVar

import numpy as np
from DashAI.back.core.schema_fields import (
    enum_field,
    int_field,
    schema_field,
    string_field,
)
from DashAI.back.core.schema_fields.base_schema import BaseSchema
from DashAI.back.core.utils import MultilingualString
from DashAI.back.models.base_model import BaseModel

from dashai_clip_model_package.device import resolve_device
from dashai_clip_model_package.hf_compat import extract_pooled_embedding
from dashai_clip_model_package.prompts import build_prompts, validate_prompt_template


class MetaCLIP2ZeroShotClassifierSchema(BaseSchema):
    model_name: schema_field(
        string_field(),
        "facebook/metaclip-2-worldwide-s16",
        alias=MultilingualString(en="Model name", es="Nombre del modelo"),
        description=MultilingualString(
            en="Hugging Face model ID for the MetaCLIP 2 checkpoint to use.",
            es="ID del modelo Hugging Face para el checkpoint MetaCLIP 2 a usar.",
        ),
    )  # type: ignore
    prompt_template: schema_field(
        string_field(),
        "a photo of a {}",
        alias=MultilingualString(en="Prompt template", es="Plantilla de prompt"),
        description=MultilingualString(
            en="Text template with exactly one '{}' label placeholder.",
            es="Plantilla de texto con exactamente un marcador de etiqueta '{}'.",
        ),
    )  # type: ignore
    batch_size: schema_field(
        int_field(ge=1),
        32,
        alias=MultilingualString(en="Batch size", es="Tamaño de lote"),
        description=MultilingualString(
            en="Number of images processed together during inference.",
            es="Número de imágenes procesadas juntas durante la inferencia.",
        ),
    )  # type: ignore
    device: schema_field(
        enum_field(enum=["auto", "cpu", "cuda"]),
        "auto",
        alias=MultilingualString(en="Device", es="Dispositivo"),
        description=MultilingualString(
            en="Compute device: automatic selection, CPU, or CUDA.",
            es="Dispositivo de cómputo: selección automática, CPU o CUDA.",
        ),
    )  # type: ignore


class MetaCLIP2ZeroShotClassifier(BaseModel):
    """Zero-shot image classifier backed by a Hugging Face MetaCLIP 2 checkpoint.

    MetaCLIP 2 uses the same joint-softmax scoring as CLIP
    (``softmax(logit_scale.exp() * cos_sim)``) and the standard CLIP
    processor, but is trained on 300+ languages, making it a multilingual
    alternative to AltCLIP for class labels or prompts outside English.
    """

    SCHEMA = MetaCLIP2ZeroShotClassifierSchema
    COMPATIBLE_COMPONENTS: ClassVar[list[str]] = ["ImageClassificationTask"]
    DISPLAY_NAME = MultilingualString(
        en="MetaCLIP 2 Zero-Shot", es="MetaCLIP 2 Zero-Shot"
    )
    DESCRIPTION = MultilingualString(
        en="Classify images by comparing them with text prompts, without fine-tuning.",
        es="Clasifica imágenes comparándolas con prompts de texto, sin ajuste fino.",
    )
    COLOR = "#3E5C76"
    ICON = "ImageSearch"

    def __init__(
        self,
        model_name="facebook/metaclip-2-worldwide-s16",
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

    def _extract_class_names(self, y_train):
        if len(y_train.column_names) != 1:
            raise ValueError(
                "MetaCLIP2ZeroShotClassifier requires exactly one output column"
            )
        column = y_train.column_names[0]
        output_type = (getattr(y_train, "types", {}) or {}).get(column)
        categories = getattr(output_type, "categories", None)
        if categories is not None:
            names = list(categories)
        else:
            names = list(dict.fromkeys(y_train[column]))
        if not names:
            raise ValueError("At least one class label is required")
        return names

    def _ensure_backend(self):
        import torch

        try:
            if self.model is None or self.processor is None:
                from transformers import CLIPProcessor, MetaClip2Model

                if self.model is None:
                    self.model = MetaClip2Model.from_pretrained(self.model_name)
                if self.processor is None:
                    self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load MetaCLIP 2 checkpoint '{self.model_name}'"
            ) from exc

    def _prepare_text_features(self):
        import torch

        prompts = build_prompts(self.class_names, self.prompt_template)
        inputs = self.processor(text=prompts, padding=True, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            text_features = extract_pooled_embedding(
                self.model.get_text_features(**inputs)
            )
        denominator = text_features.norm(p=2, dim=-1, keepdim=True).clamp_min(
            torch.finfo(text_features.dtype).eps
        )
        self._text_features = text_features / denominator

    def train(self, x_train, y_train, x_validation=None, y_validation=None):
        self.class_names = self._extract_class_names(y_train)
        self.label_to_idx = {
            label: index for index, label in enumerate(self.class_names)
        }
        self.idx_to_label = {index: label for label, index in self.label_to_idx.items()}
        self._text_features = None
        self._ensure_backend()
        self._prepare_text_features()
        return self

    def prepare_output(self, dataset, is_fit=False):
        if not self.label_to_idx:
            raise RuntimeError(
                "MetaCLIP2ZeroShotClassifier class labels are not initialized"
            )
        if len(dataset.column_names) != 1:
            raise ValueError(
                "MetaCLIP2ZeroShotClassifier requires exactly one output column"
            )
        column = dataset.column_names[0]
        try:
            encoded = [self.label_to_idx[value] for value in dataset[column]]
        except KeyError as exc:
            raise ValueError(f"Unknown class label: {exc.args[0]!r}") from exc
        import pyarrow as pa
        from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset

        return DashAIDataset(pa.table({column: encoded}))

    def _to_pil_image(self, value, index):
        try:
            return value.to_pil().convert("RGB")
        except Exception as exc:
            raise ValueError(f"Unable to decode image at sample {index}") from exc

    def predict(self, x):
        import torch

        if not self.class_names:
            raise RuntimeError(
                "MetaCLIP2ZeroShotClassifier must be trained or loaded before "
                "prediction; call train or load first"
            )
        if len(x.column_names) != 1:
            raise ValueError(
                "MetaCLIP2ZeroShotClassifier requires exactly one input column"
            )

        if len(x) == 0:
            return np.empty((0, len(self.class_names)), dtype=np.float32)

        try:
            self._ensure_backend()
            if self._text_features is None:
                self._prepare_text_features()

            column = x.column_names[0]
            values = x[column]
            batches = []
            for start in range(0, len(x), self.batch_size):
                images = [
                    self._to_pil_image(value, index)
                    for index, value in enumerate(
                        values[start : start + self.batch_size], start
                    )
                ]
                image_inputs = self.processor(images=images, return_tensors="pt")
                image_inputs = {
                    name: value.to(self.device) for name, value in image_inputs.items()
                }
                with torch.inference_mode():
                    image_features = extract_pooled_embedding(
                        self.model.get_image_features(**image_inputs)
                    )
                    denominator = image_features.norm(dim=-1, keepdim=True).clamp_min(
                        torch.finfo(image_features.dtype).eps
                    )
                    image_features = image_features / denominator
                    logits = (
                        self.model.logit_scale.exp()
                        * image_features
                        @ self._text_features.T
                    )
                    batches.append(
                        torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
                    )
            return np.concatenate(batches, axis=0)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "CUDA out of memory during MetaCLIP 2 inference; "
                f"try reducing batch_size (currently {self.batch_size})"
            ) from exc

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
            "format_version",
            "model_name",
            "prompt_template",
            "batch_size",
            "device_name",
            "class_names",
        }
        if not isinstance(state, dict) or required - state.keys():
            raise ValueError("Invalid MetaCLIP2ZeroShotClassifier checkpoint")
        if state["format_version"] != 1:
            raise ValueError(
                "Unsupported MetaCLIP2ZeroShotClassifier checkpoint version"
            )
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
