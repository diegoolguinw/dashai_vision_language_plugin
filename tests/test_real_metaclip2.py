import os

import pytest

from dashai_clip_model_package import MetaCLIP2ZeroShotClassifier


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_CLIP_INTEGRATION") != "1",
    reason="set RUN_CLIP_INTEGRATION=1 to download and run MetaCLIP 2",
)
def test_real_metaclip2_smoke(fake_two_class_dashai_dataset):
    x, y = fake_two_class_dashai_dataset
    component = MetaCLIP2ZeroShotClassifier(batch_size=2, device="cpu")
    component.train(x, y)
    probabilities = component.predict(x)
    assert probabilities.shape == (2, 2)
