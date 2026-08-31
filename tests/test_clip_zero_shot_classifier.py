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
