import sys
import types

import numpy as np
import pytest
import torch
from conftest import BatchEncoding, DeviceTrackingTensor, FakeDataset, FakeImage

from dashai_vision_language_plugin.align_zero_shot_classifier import (
    ALIGNZeroShotClassifier,
    ALIGNZeroShotClassifierSchema,
)


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
    component = ALIGNZeroShotClassifier(device="cpu")

    assert component.model is None
    assert component.processor is None
    assert component.device.type == "cpu"
    assert component.COMPATIBLE_COMPONENTS == ["ImageClassificationTask"]
    assert component.SCHEMA is ALIGNZeroShotClassifierSchema


def test_train_preserves_categorical_order(monkeypatch, categorical_type):
    component = ALIGNZeroShotClassifier(device="cpu")
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
    component = ALIGNZeroShotClassifier(device="cpu")
    monkeypatch.setattr(component, "_ensure_backend", lambda: None)
    monkeypatch.setattr(component, "_prepare_text_features", lambda: None)
    y = FakeDataset("label", ["bird", "cat", "bird", "dog"])

    component.train(FakeDataset("image", []), y)

    assert component.class_names == ["bird", "cat", "dog"]


def test_train_prepares_text_without_gradients_or_parameter_changes():
    component = ALIGNZeroShotClassifier(device="cpu")
    component.model = RecordingModel()
    component.processor = RecordingProcessor()
    before = [parameter.detach().clone() for parameter in component.model.parameters()]
    flags = [parameter.requires_grad for parameter in component.model.parameters()]

    component.train(FakeDataset("image", []), FakeDataset("label", ["cat", "dog"]))

    assert component.model.training is False
    assert component.model.text_grad_enabled is False
    assert all(
        tensor.devices == [torch.device("cpu")]
        for tensor in component.processor.tensors
    )
    assert all(
        value.device.type == "cpu" for value in component.model.text_inputs.values()
    )
    assert torch.allclose(
        component._text_features, torch.tensor([[0.6, 0.8], [0.0, 0.0]])
    )
    after = list(component.model.parameters())
    assert all(torch.equal(old, new) for old, new in zip(before, after))
    assert [parameter.requires_grad for parameter in after] == flags


def test_train_unwraps_pooled_text_output_from_newer_transformers():
    class PoolingRecordingModel(RecordingModel):
        def get_text_features(self, **inputs):
            from types import SimpleNamespace

            return SimpleNamespace(pooler_output=super().get_text_features(**inputs))

    component = ALIGNZeroShotClassifier(device="cpu")
    component.model = PoolingRecordingModel()
    component.processor = RecordingProcessor()

    component.train(FakeDataset("image", []), FakeDataset("label", ["cat", "dog"]))

    assert torch.allclose(
        component._text_features, torch.tensor([[0.6, 0.8], [0.0, 0.0]])
    )


def test_ensure_backend_loads_checkpoint_lazily(monkeypatch):
    model = RecordingModel()
    processor = RecordingProcessor()

    class FakeAlignModel:
        @staticmethod
        def from_pretrained(model_name):
            model.checkpoint = model_name
            return model

    class FakeAlignProcessor:
        @staticmethod
        def from_pretrained(model_name):
            processor.checkpoint = model_name
            return processor

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AlignModel = FakeAlignModel
    fake_transformers.AlignProcessor = FakeAlignProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    component = ALIGNZeroShotClassifier(model_name="test/checkpoint", device="cpu")

    component._ensure_backend()

    assert component.model is model
    assert component.processor is processor
    assert component.model.device == torch.device("cpu")
    assert component.model.training is False
    assert component.model.checkpoint == "test/checkpoint"
    assert component.processor.checkpoint == "test/checkpoint"


def test_ensure_backend_includes_checkpoint_in_load_error(monkeypatch):
    class FailingAlignModel:
        @staticmethod
        def from_pretrained(model_name):
            raise OSError("not available")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AlignModel = FailingAlignModel
    fake_transformers.AlignProcessor = object()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    component = ALIGNZeroShotClassifier(model_name="missing/checkpoint", device="cpu")

    with pytest.raises(
        RuntimeError, match="Unable to load ALIGN checkpoint 'missing/checkpoint'"
    ):
        component._ensure_backend()


def test_prepare_output_encodes_labels_in_canonical_order():
    from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset

    component = ALIGNZeroShotClassifier(device="cpu")
    component.label_to_idx = {"dog": 0, "cat": 1}

    output = component.prepare_output(FakeDataset("label", ["cat", "dog"]))

    assert isinstance(output, DashAIDataset)
    assert output["label"] == [1, 0]


def test_prepare_output_rejects_untrained_class_mapping():
    component = ALIGNZeroShotClassifier(device="cpu")

    with pytest.raises(RuntimeError, match="class labels are not initialized"):
        component.prepare_output(FakeDataset("label", ["cat"]))


def test_prepare_output_requires_exactly_one_column():
    component = ALIGNZeroShotClassifier(device="cpu")
    component.label_to_idx = {"cat": 0}
    dataset = FakeDataset("label", ["cat"])
    dataset.column_names = ["label", "other"]

    with pytest.raises(ValueError, match="exactly one output column"):
        component.prepare_output(dataset)


def test_prepare_output_rejects_unknown_class_label():
    component = ALIGNZeroShotClassifier(device="cpu")
    component.label_to_idx = {"cat": 0}

    with pytest.raises(ValueError, match="Unknown class label: 'dog'"):
        component.prepare_output(FakeDataset("label", ["dog"]))


def test_extract_class_names_rejects_declared_empty_categories():
    component = ALIGNZeroShotClassifier(device="cpu")
    empty_categories = types.SimpleNamespace(categories=[])
    y = FakeDataset("label", ["cat"], empty_categories)

    with pytest.raises(ValueError, match="At least one class label is required"):
        component._extract_class_names(y)


def test_extract_class_names_rejects_no_observed_labels():
    component = ALIGNZeroShotClassifier(device="cpu")

    with pytest.raises(ValueError, match="At least one class label is required"):
        component._extract_class_names(FakeDataset("label", []))


def _fitted_component(fake_align_backend, batch_size=2):
    component = ALIGNZeroShotClassifier(batch_size=batch_size, device="cpu")
    component.model = fake_align_backend.model
    component.processor = fake_align_backend.processor
    component.class_names = ["red", "blue"]
    component.label_to_idx = {"red": 0, "blue": 1}
    component.idx_to_label = {0: "red", 1: "blue"}
    component._text_features = fake_align_backend.text_features
    return component


def test_predict_returns_ordered_probabilities_in_batches(fake_align_backend):
    component = _fitted_component(fake_align_backend)
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


def test_predict_unwraps_pooled_output_from_newer_transformers(
    fake_align_backend_with_pooling,
):
    component = _fitted_component(fake_align_backend_with_pooling)
    dataset = FakeDataset(
        "image",
        [FakeImage("red"), FakeImage("blue"), FakeImage("red")],
    )

    probabilities = component.predict(dataset)

    assert probabilities.shape == (3, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert probabilities.argmax(axis=1).tolist() == [0, 1, 0]


def test_predict_applies_temperature_division_softmax_and_inference_mode(
    fake_align_backend,
):
    component = _fitted_component(fake_align_backend)
    with torch.no_grad():
        component.model.temperature.fill_(0.5)
    observed = {}

    def non_unit_features(pixel_values):
        observed["grad_enabled"] = torch.is_grad_enabled()
        observed["device"] = pixel_values.device
        return torch.tensor([[3.0, 4.0], [0.0, 5.0]], device=pixel_values.device)

    component.model.get_image_features = non_unit_features

    probabilities = component.predict(
        FakeDataset("image", [FakeImage("red"), FakeImage("blue")])
    )

    np.testing.assert_allclose(
        probabilities,
        np.array([[0.40131235, 0.59868765], [0.11920292, 0.88079708]]),
        atol=1e-6,
    )
    assert observed == {"grad_enabled": False, "device": torch.device("cpu")}
    assert isinstance(component.processor.image_encodings[0], BatchEncoding)
    assert component.processor.image_tensors[0].devices == [torch.device("cpu")]


def test_predict_requires_training_or_loading_class_setup():
    component = ALIGNZeroShotClassifier(device="cpu")

    with pytest.raises(RuntimeError, match="train or load"):
        component.predict(FakeDataset("image", [FakeImage("red")]))


def test_predict_returns_empty_float32_array_for_empty_input(fake_align_backend):
    component = _fitted_component(fake_align_backend)

    probabilities = component.predict(FakeDataset("image", []))

    assert probabilities.shape == (0, 2)
    assert probabilities.dtype == np.float32
    assert component.processor.image_batch_sizes == []


def test_predict_empty_input_does_not_initialize_the_backend():
    class UntouchedModel:
        def __init__(self):
            self.to_calls = 0

        def to(self, _device):
            self.to_calls += 1
            raise AssertionError("empty input must not initialize the model")

    component = ALIGNZeroShotClassifier(device="cpu")
    component.class_names = ["red", "blue"]
    component.model = UntouchedModel()
    component.processor = object()

    probabilities = component.predict(FakeDataset("image", []))

    assert probabilities.shape == (0, 2)
    assert component.model.to_calls == 0


def test_predict_identifies_the_undecodable_image_sample(fake_align_backend):
    component = _fitted_component(fake_align_backend)

    with pytest.raises(ValueError, match="Unable to decode image at sample 1"):
        component.predict(FakeDataset("image", [FakeImage("red"), object()]))


def test_predict_converts_images_to_rgb(fake_align_backend):
    component = _fitted_component(fake_align_backend)

    component.predict(FakeDataset("image", [FakeImage(127, mode="L")]))

    assert component.processor.image_batches[0][0].mode == "RGB"


def test_predict_chains_the_original_image_decode_error(fake_align_backend):
    class BrokenImage:
        def to_pil(self):
            raise OSError("corrupt image")

    component = _fitted_component(fake_align_backend)

    with pytest.raises(ValueError, match="Unable to decode image at sample 0") as error:
        component.predict(FakeDataset("image", [BrokenImage()]))

    assert isinstance(error.value.__cause__, OSError)


def test_predict_requires_exactly_one_input_column(fake_align_backend):
    component = _fitted_component(fake_align_backend)
    dataset = FakeDataset("image", [FakeImage("red")])
    dataset.column_names = ["image", "other"]

    with pytest.raises(
        ValueError,
        match="ALIGNZeroShotClassifier requires exactly one input column",
    ):
        component.predict(dataset)


def test_predict_adds_a_batch_size_hint_to_cuda_out_of_memory(fake_align_backend):
    component = _fitted_component(fake_align_backend)

    def raise_out_of_memory(**_inputs):
        raise torch.cuda.OutOfMemoryError("simulated CUDA OOM")

    component.model.get_image_features = raise_out_of_memory

    with pytest.raises(RuntimeError, match="batch_size") as error:
        component.predict(FakeDataset("image", [FakeImage("red")]))

    assert isinstance(error.value.__cause__, torch.cuda.OutOfMemoryError)


def test_predict_adds_a_batch_size_hint_to_backend_cuda_out_of_memory(
    fake_align_backend,
):
    component = _fitted_component(fake_align_backend)

    def raise_out_of_memory(_device):
        raise torch.cuda.OutOfMemoryError("simulated backend CUDA OOM")

    component.model.to = raise_out_of_memory

    with pytest.raises(RuntimeError, match="batch_size") as error:
        component.predict(FakeDataset("image", [FakeImage("red")]))

    assert isinstance(error.value.__cause__, torch.cuda.OutOfMemoryError)


def test_predict_adds_a_batch_size_hint_to_text_feature_cuda_out_of_memory(
    fake_align_backend,
):
    component = _fitted_component(fake_align_backend)
    component._text_features = None

    def raise_out_of_memory(**_inputs):
        raise torch.cuda.OutOfMemoryError("simulated text CUDA OOM")

    component.model.get_text_features = raise_out_of_memory

    with pytest.raises(RuntimeError, match="batch_size") as error:
        component.predict(FakeDataset("image", [FakeImage("red")]))

    assert isinstance(error.value.__cause__, torch.cuda.OutOfMemoryError)


def test_save_load_round_trip_is_lazy(tmp_path):
    component = ALIGNZeroShotClassifier(
        model_name="org/checkpoint",
        prompt_template="an image of {}",
        batch_size=7,
        device="cpu",
    )
    component.class_names = ["dog", "cat"]
    component.label_to_idx = {"dog": 0, "cat": 1}
    component.idx_to_label = {0: "dog", 1: "cat"}
    component.model = object()
    component.processor = object()
    path = tmp_path / "clip.pt"

    component.save(path)
    restored = ALIGNZeroShotClassifier.load(path)

    assert restored.model_name == "org/checkpoint"
    assert restored.prompt_template == "an image of {}"
    assert restored.batch_size == 7
    assert restored.device_name == "cpu"
    assert restored.class_names == ["dog", "cat"]
    assert restored.label_to_idx == {"dog": 0, "cat": 1}
    assert restored.idx_to_label == {0: "dog", 1: "cat"}
    assert restored.model is None
    assert restored.processor is None
    assert restored._text_features is None


def test_load_rejects_malformed_checkpoint(tmp_path):
    path = tmp_path / "clip.pt"
    torch.save({"format_version": 1}, path)

    with pytest.raises(ValueError, match="Invalid ALIGNZeroShotClassifier checkpoint"):
        ALIGNZeroShotClassifier.load(path)


def test_load_rejects_unsupported_checkpoint_version(tmp_path):
    path = tmp_path / "clip.pt"
    torch.save(
        {
            "format_version": 2,
            "model_name": "org/checkpoint",
            "prompt_template": "an image of {}",
            "batch_size": 7,
            "device_name": "cpu",
            "class_names": ["dog", "cat"],
        },
        path,
    )

    with pytest.raises(
        ValueError, match="Unsupported ALIGNZeroShotClassifier checkpoint version"
    ):
        ALIGNZeroShotClassifier.load(path)
