import os

import pytest

from dashai_vision_language_plugin import AltCLIPZeroShotClassifier


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_CLIP_INTEGRATION") != "1",
    reason="set RUN_CLIP_INTEGRATION=1 to download and run AltCLIP",
)
def test_real_altclip_smoke(fake_two_class_dashai_dataset):
    x, y = fake_two_class_dashai_dataset
    component = AltCLIPZeroShotClassifier(batch_size=2, device="cpu")
    component.train(x, y)
    probabilities = component.predict(x)
    assert probabilities.shape == (2, 2)
