import torch
from conftest import FakeDataset, FakeImage

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


def test_train_preserves_categorical_order(monkeypatch, categorical_type):
    component = CLIPZeroShotClassifier(device="cpu")
    monkeypatch.setattr(component, "_ensure_backend", lambda: None)
    monkeypatch.setattr(component, "_prepare_text_features", lambda: None)
    x = FakeDataset("image", [FakeImage("red"), FakeImage("blue")])
    y = FakeDataset("label", ["cat", "dog"], categorical_type)

    result = component.train(x, y)

    assert result is component
    assert component.class_names == ["dog", "cat"]
    assert component.label_to_idx == {"dog": 0, "cat": 1}
    assert component.idx_to_label == {0: "dog", 1: "cat"}


def test_train_fallback_uses_first_seen_order(monkeypatch):
    component = CLIPZeroShotClassifier(device="cpu")
    monkeypatch.setattr(component, "_ensure_backend", lambda: None)
    monkeypatch.setattr(component, "_prepare_text_features", lambda: None)
    y = FakeDataset("label", ["bird", "cat", "bird", "dog"])

    component.train(FakeDataset("image", []), y)

    assert component.class_names == ["bird", "cat", "dog"]


def test_train_does_not_modify_model_parameters(monkeypatch):
    component = CLIPZeroShotClassifier(device="cpu")
    component.model = torch.nn.Linear(2, 2)
    before = [parameter.detach().clone() for parameter in component.model.parameters()]
    flags = [parameter.requires_grad for parameter in component.model.parameters()]
    monkeypatch.setattr(component, "_ensure_backend", lambda: None)
    monkeypatch.setattr(component, "_prepare_text_features", lambda: None)
    y = FakeDataset("label", ["cat", "dog"])

    component.train(FakeDataset("image", []), y)

    after = list(component.model.parameters())
    assert all(torch.equal(old, new) for old, new in zip(before, after))
    assert [parameter.requires_grad for parameter in after] == flags


def test_prepare_output_encodes_labels_in_canonical_order():
    component = CLIPZeroShotClassifier(device="cpu")
    component.label_to_idx = {"dog": 0, "cat": 1}

    output = component.prepare_output(FakeDataset("label", ["cat", "dog"]))

    assert output["label"] == [1, 0]
