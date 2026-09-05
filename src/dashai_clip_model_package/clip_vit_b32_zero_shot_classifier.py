from DashAI.back.core.utils import MultilingualString

from dashai_clip_model_package.clip_zero_shot_classifier_base import (
    CLIPZeroShotClassifierBase,
    CLIPZeroShotClassifierSchema,
)


class CLIPViTB32ZeroShotClassifier(CLIPZeroShotClassifierBase):
    """CLIP ViT-B/32 zero-shot image classifier. The fastest and smallest
    CLIP checkpoint offered by this plugin; a good default."""

    SCHEMA = CLIPZeroShotClassifierSchema
    MODEL_NAME = "openai/clip-vit-base-patch32"
    DISPLAY_NAME = MultilingualString(
        en="CLIP ViT-B/32 Zero-Shot", es="CLIP ViT-B/32 Zero-Shot"
    )
    DESCRIPTION = MultilingualString(
        en="Classify images by comparing them with text prompts, without "
        "fine-tuning. Uses the smallest, fastest CLIP checkpoint "
        "(ViT-B/32) — a good default.",
        es="Clasifica imágenes comparándolas con prompts de texto, sin "
        "ajuste fino. Usa el checkpoint CLIP más pequeño y rápido "
        "(ViT-B/32) — una buena opción por defecto.",
    )
    COLOR = "#5B4BDB"
    ICON = "ImageSearch"
