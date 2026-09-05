from DashAI.back.core.utils import MultilingualString

from dashai_vision_language_plugin.clip_zero_shot_classifier_base import (
    CLIPZeroShotClassifierBase,
    CLIPZeroShotClassifierSchema,
)


class CLIPViTB16ZeroShotClassifier(CLIPZeroShotClassifierBase):
    """CLIP ViT-B/16 zero-shot image classifier. A mid-sized CLIP
    checkpoint: more accurate than ViT-B/32, smaller and faster than
    ViT-L/14."""

    SCHEMA = CLIPZeroShotClassifierSchema
    MODEL_NAME = "openai/clip-vit-base-patch16"
    DISPLAY_NAME = MultilingualString(
        en="CLIP ViT-B/16 Zero-Shot", es="CLIP ViT-B/16 Zero-Shot"
    )
    DESCRIPTION = MultilingualString(
        en="Classify images by comparing them with text prompts, without "
        "fine-tuning. Uses a mid-sized CLIP checkpoint (ViT-B/16): more "
        "accurate than ViT-B/32, lighter than ViT-L/14.",
        es="Clasifica imágenes comparándolas con prompts de texto, sin "
        "ajuste fino. Usa un checkpoint CLIP de tamaño intermedio "
        "(ViT-B/16): más preciso que ViT-B/32, más liviano que ViT-L/14.",
    )
    COLOR = "#4B3AB0"
    ICON = "ImageSearch"
