import sys
import types

import numpy as np
import pytest
import torch
from conftest import FakeDataset, FakeImage

from dashai_clip_model_package.clip_zero_shot_classifier import (
    CLIPZeroShotClassifier,
    CLIPZeroShotClassifierSchema,
)


class DeviceTrackingTensor:
    def __init__(self, tensor):
        self.tensor = tensor
        self.devices = []

    def to(self, device):
        self.devices.append(torch.device(device))
        return self.tensor.to(device)


class RecordingProcessor:
    def __init__(self):
        self.inputs = None
        self.tensors = []

    def __call__(self, **inputs):
        self.inputs = inputs
        self.tensors = [
            DeviceTrackingTensor(torch.tensor([[1, 2], [3, 4]])),
            DeviceTrackingTensor(torch.tensor([[1, 1], [1, 1]])),
        ]
        return {
            "input_ids": self.tensors[0],
            "attention_mask": self.tensors[1],
        }


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.device = None
        self.text_grad_enabled = None
        self.text_inputs = None

    def to(self, device):
        self.device = torch.device(device)
        return super().to(device)

    def get_text_features(self, **inputs):
        self.text_grad_enabled = torch.is_grad_enabled()
        self.text_inputs = inputs
        return torch.tensor([[3.0, 4.0], [0.0, 0.0]], device=self.weight.device)


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


def test_train_prepares_text_without_gradients_or_parameter_changes():
    component = CLIPZeroShotClassifier(device="cpu")
    component.model = RecordingModel()
    component.processor = RecordingProcessor()
    before = [parameter.detach().clone() for parameter in component.model.parameters()]
    flags = [parameter.requires_grad for parameter in component.model.parameters()]

    component.train(
        FakeDataset("image", []), FakeDataset("label", ["cat", "dog"])
    )

    assert component.model.training is False
    assert component.model.text_grad_enabled is False
    assert all(tensor.devices == [torch.device("cpu")] for tensor in component.processor.tensors)
    assert all(value.device.type == "cpu" for value in component.model.text_inputs.values())
    assert torch.allclose(
        component._text_features, torch.tensor([[0.6, 0.8], [0.0, 0.0]])
    )
    after = list(component.model.parameters())
    assert all(torch.equal(old, new) for old, new in zip(before, after))
    assert [parameter.requires_grad for parameter in after] == flags


def test_ensure_backend_loads_checkpoint_lazily(monkeypatch):
    model = RecordingModel()
    processor = RecordingProcessor()

    class FakeCLIPModel:
        @staticmethod
        def from_pretrained(model_name):
            model.checkpoint = model_name
            return model

    class FakeCLIPProcessor:
        @staticmethod
        def from_pretrained(model_name):
            processor.checkpoint = model_name
            return processor

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.CLIPModel = FakeCLIPModel
    fake_transformers.CLIPProcessor = FakeCLIPProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    component = CLIPZeroShotClassifier(model_name="test/checkpoint", device="cpu")

    component._ensure_backend()

    assert component.model is model
    assert component.processor is processor
    assert component.model.device == torch.device("cpu")
    assert component.model.training is False
    assert component.model.checkpoint == "test/checkpoint"
    assert component.processor.checkpoint == "test/checkpoint"


def test_ensure_backend_includes_checkpoint_in_load_error(monkeypatch):
    class FailingCLIPModel:
        @staticmethod
        def from_pretrained(model_name):
            raise OSError("not available")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.CLIPModel = FailingCLIPModel
    fake_transformers.CLIPProcessor = object()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    component = CLIPZeroShotClassifier(model_name="missing/checkpoint", device="cpu")

    with pytest.raises(
        RuntimeError, match="Unable to load CLIP checkpoint 'missing/checkpoint'"
    ):
        component._ensure_backend()


def test_prepare_output_encodes_labels_in_canonical_order():
    from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset

    component = CLIPZeroShotClassifier(device="cpu")
    component.label_to_idx = {"dog": 0, "cat": 1}

    output = component.prepare_output(FakeDataset("label", ["cat", "dog"]))

    assert isinstance(output, DashAIDataset)
    assert output["label"] == [1, 0]


def test_prepare_output_rejects_untrained_class_mapping():
    component = CLIPZeroShotClassifier(device="cpu")

    with pytest.raises(RuntimeError, match="class labels are not initialized"):
        component.prepare_output(FakeDataset("label", ["cat"]))


def test_prepare_output_requires_exactly_one_column():
    component = CLIPZeroShotClassifier(device="cpu")
    component.label_to_idx = {"cat": 0}
    dataset = FakeDataset("label", ["cat"])
    dataset.column_names = ["label", "other"]

    with pytest.raises(ValueError, match="exactly one output column"):
        component.prepare_output(dataset)


def test_prepare_output_rejects_unknown_class_label():
    component = CLIPZeroShotClassifier(device="cpu")
    component.label_to_idx = {"cat": 0}

    with pytest.raises(ValueError, match="Unknown class label: 'dog'"):
        component.prepare_output(FakeDataset("label", ["dog"]))


def test_extract_class_names_rejects_declared_empty_categories():
    component = CLIPZeroShotClassifier(device="cpu")
    empty_categories = types.SimpleNamespace(categories=[])
    y = FakeDataset("label", ["cat"], empty_categories)

    with pytest.raises(ValueError, match="At least one class label is required"):
        component._extract_class_names(y)


def test_extract_class_names_rejects_no_observed_labels():
    component = CLIPZeroShotClassifier(device="cpu")

    with pytest.raises(ValueError, match="At least one class label is required"):
        component._extract_class_names(FakeDataset("label", []))


def _fitted_component(fake_backend, batch_size=2):
    component = CLIPZeroShotClassifier(batch_size=batch_size, device="cpu")
    component.model = fake_backend.model
    component.processor = fake_backend.processor
    component.class_names = ["red", "blue"]
    component.label_to_idx = {"red": 0, "blue": 1}
    component.idx_to_label = {0: "red", 1: "blue"}
    component._text_features = fake_backend.text_features
    return component


def test_predict_returns_ordered_probabilities_in_batches(fake_backend):
    component = _fitted_component(fake_backend)
    dataset = FakeDataset(
        "image",
        [FakeImage("red"), FakeImage("blue"), FakeImage("red")],
    )

    probabilities = component.predict(dataset)

    assert probabilities.shape == (3, 2)
    assert probabilities.dtype == np.float32
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(probabilities >= 0)
    assert probabilities.argmax(axis=1).tolist() == [0, 1, 0]
    assert component.processor.image_batch_sizes == [2, 1]


def test_predict_requires_training_or_loading_class_setup():
    component = CLIPZeroShotClassifier(device="cpu")

    with pytest.raises(RuntimeError, match="train or load"):
        component.predict(FakeDataset("image", [FakeImage("red")]))


def test_predict_returns_empty_float32_array_for_empty_input(fake_backend):
    component = _fitted_component(fake_backend)

    probabilities = component.predict(FakeDataset("image", []))

    assert probabilities.shape == (0, 2)
    assert probabilities.dtype == np.float32
    assert component.processor.image_batch_sizes == []


def test_predict_identifies_the_undecodable_image_sample(fake_backend):
    component = _fitted_component(fake_backend)

    with pytest.raises(ValueError, match="Unable to decode image at sample 1"):
        component.predict(FakeDataset("image", [FakeImage("red"), object()]))


def test_predict_requires_exactly_one_input_column(fake_backend):
    component = _fitted_component(fake_backend)
    dataset = FakeDataset("image", [FakeImage("red")])
    dataset.column_names = ["image", "other"]

    with pytest.raises(
        ValueError,
        match="CLIPZeroShotClassifier requires exactly one input column",
    ):
        component.predict(dataset)


def test_predict_adds_a_batch_size_hint_to_cuda_out_of_memory(fake_backend):
    component = _fitted_component(fake_backend)

    def raise_out_of_memory(**_inputs):
        raise torch.cuda.OutOfMemoryError("simulated CUDA OOM")

    component.model.get_image_features = raise_out_of_memory

    with pytest.raises(RuntimeError, match="batch_size") as error:
        component.predict(FakeDataset("image", [FakeImage("red")]))

    assert isinstance(error.value.__cause__, torch.cuda.OutOfMemoryError)
