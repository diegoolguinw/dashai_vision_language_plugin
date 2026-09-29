import pytest

from dashai_vision_language_plugin.clip_vit_b16_zero_shot_classifier import (
    CLIPViTB16ZeroShotClassifier,
)
from dashai_vision_language_plugin.clip_vit_b32_zero_shot_classifier import (
    CLIPViTB32ZeroShotClassifier,
)
from dashai_vision_language_plugin.clip_vit_l14_zero_shot_classifier import (
    CLIPViTL14ZeroShotClassifier,
)
from dashai_vision_language_plugin.clip_zero_shot_classifier_base import (
    CLIPZeroShotClassifier,
)

VARIANTS = [
    (CLIPViTB32ZeroShotClassifier, "openai/clip-vit-base-patch32"),
    (CLIPViTB16ZeroShotClassifier, "openai/clip-vit-base-patch16"),
    (CLIPViTL14ZeroShotClassifier, "openai/clip-vit-large-patch14"),
]


@pytest.mark.parametrize("component_class, expected_model_name", VARIANTS)
def test_variant_has_fixed_checkpoint_and_no_model_name_parameter(
    component_class, expected_model_name
):
    component = component_class(device="cpu")

    assert issubclass(component_class, CLIPZeroShotClassifier)
    assert component.MODEL_NAME == expected_model_name
    assert component.model_name == expected_model_name
    assert "model_name" not in component.SCHEMA.model_fields


@pytest.mark.parametrize("component_class, _expected_model_name", VARIANTS)
def test_variant_has_a_distinct_display_name(component_class, _expected_model_name):
    assert component_class.DISPLAY_NAME.en
    assert component_class.DISPLAY_NAME.es


def test_variants_have_distinct_checkpoints():
    checkpoints = {model_name for _cls, model_name in VARIANTS}
    assert len(checkpoints) == len(VARIANTS)
