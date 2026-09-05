from DashAI.back.core.utils import MultilingualString

from dashai_clip_model_package.clip_zero_shot_classifier_base import (
    CLIPZeroShotClassifierBase,
    CLIPZeroShotClassifierSchema,
)


class CLIPViTL14ZeroShotClassifier(CLIPZeroShotClassifierBase):
    """CLIP ViT-L/14 zero-shot image classifier. The largest, most
    accurate CLIP checkpoint offered by this plugin; slower and heavier
    to download than the ViT-B variants."""

    SCHEMA = CLIPZeroShotClassifierSchema
    MODEL_NAME = "openai/clip-vit-large-patch14"
    DISPLAY_NAME = MultilingualString(
        en="CLIP ViT-L/14 Zero-Shot", es="CLIP ViT-L/14 Zero-Shot"
    )
    DESCRIPTION = MultilingualString(
        en="Classify images by comparing them with text prompts, without "
        "fine-tuning. Uses the largest, most accurate CLIP checkpoint "
        "(ViT-L/14) — slower and heavier to download than the ViT-B "
        "variants.",
        es="Clasifica imágenes comparándolas con prompts de texto, sin "
        "ajuste fino. Usa el checkpoint CLIP más grande y preciso "
        "(ViT-L/14) — más lento y pesado de descargar que las variantes "
        "ViT-B.",
    )
    COLOR = "#3A2C87"
    ICON = "ImageSearch"
