from typing import ClassVar

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
from dashai_clip_model_package.prompts import validate_prompt_template


class CLIPZeroShotClassifierSchema(BaseSchema):
    model_name: schema_field(
        string_field(),
        "openai/clip-vit-base-patch32",
        alias=MultilingualString(en="Model name", es="Nombre del modelo"),
        description=MultilingualString(
            en="Hugging Face model ID for the CLIP checkpoint to use.",
            es="ID del modelo Hugging Face para el checkpoint CLIP a usar.",
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
        alias=MultilingualString(en="Batch size", es="Tama\u00f1o de lote"),
        description=MultilingualString(
            en="Number of images processed together during inference.",
            es="N\u00famero de im\u00e1genes procesadas juntas durante la inferencia.",
        ),
    )  # type: ignore
    device: schema_field(
        enum_field(enum=["auto", "cpu", "cuda"]),
        "auto",
        alias=MultilingualString(en="Device", es="Dispositivo"),
        description=MultilingualString(
            en="Compute device: automatic selection, CPU, or CUDA.",
            es="Dispositivo de c\u00f3mputo: selecci\u00f3n autom\u00e1tica, CPU o CUDA.",
        ),
    )  # type: ignore


class CLIPZeroShotClassifier(BaseModel):
    SCHEMA = CLIPZeroShotClassifierSchema
    COMPATIBLE_COMPONENTS: ClassVar[list[str]] = ["ImageClassificationTask"]
    DISPLAY_NAME = MultilingualString(en="CLIP Zero-Shot", es="CLIP Zero-Shot")
    DESCRIPTION = MultilingualString(
        en="Classify images by comparing them with text prompts, without fine-tuning.",
        es="Clasifica im\u00e1genes compar\u00e1ndolas con prompts de texto, sin ajuste fino.",
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
