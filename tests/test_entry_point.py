from importlib.metadata import entry_points


def test_public_component_is_exported():
    from dashai_clip_model_package import CLIPZeroShotClassifier

    assert CLIPZeroShotClassifier.__name__ == "CLIPZeroShotClassifier"


def test_dashai_entry_point_loads_component():
    matches = [
        point
        for point in entry_points(group="dashai.plugins")
        if point.name == "CLIPZeroShotClassifier"
    ]
    assert len(matches) == 1
    assert matches[0].load().__name__ == "CLIPZeroShotClassifier"
