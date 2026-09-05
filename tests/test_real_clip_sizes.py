import os

import pytest

from dashai_clip_model_package import (
    CLIPViTB16ZeroShotClassifier,
    CLIPViTB32ZeroShotClassifier,
    CLIPViTL14ZeroShotClassifier,
)


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_CLIP_INTEGRATION") != "1",
    reason="set RUN_CLIP_INTEGRATION=1 to download and run CLIP",
)
@pytest.mark.parametrize(
    "component_class",
    [
        CLIPViTB32ZeroShotClassifier,
        CLIPViTB16ZeroShotClassifier,
        CLIPViTL14ZeroShotClassifier,
    ],
)
def test_real_clip_size_variant_smoke(component_class, fake_two_class_dashai_dataset):
    x, y = fake_two_class_dashai_dataset
    component = component_class(batch_size=2, device="cpu")
    component.train(x, y)
    probabilities = component.predict(x)
    assert probabilities.shape == (2, 2)
