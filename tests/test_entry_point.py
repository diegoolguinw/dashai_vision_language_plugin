from importlib.metadata import entry_points

import pytest

COMPONENT_NAMES = [
    "CLIPViTB32ZeroShotClassifier",
    "CLIPViTB16ZeroShotClassifier",
    "CLIPViTL14ZeroShotClassifier",
    "SigLIPZeroShotClassifier",
    "ALIGNZeroShotClassifier",
    "AltCLIPZeroShotClassifier",
    "MetaCLIP2ZeroShotClassifier",
]


def test_public_component_is_exported():
    import dashai_clip_model_package

    for name in COMPONENT_NAMES:
        component = getattr(dashai_clip_model_package, name)
        assert component.__name__ == name


@pytest.mark.parametrize("component_name", COMPONENT_NAMES)
def test_dashai_entry_point_loads_component(component_name):
    matches = [
        point
        for point in entry_points(group="dashai.plugins")
        if point.name == component_name
    ]
    assert len(matches) == 1
    assert matches[0].load().__name__ == component_name
