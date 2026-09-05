from types import SimpleNamespace

import pytest
import torch
from PIL import Image


class BatchEncoding(dict):
    pass


class FakeImage:
    def __init__(self, color, mode="RGB"):
        self.color = color
        self.mode = mode

    def to_pil(self):
        return Image.new(self.mode, (2, 2), self.color)


class FakeDataset:
    def __init__(self, column, values, column_type=None):
        self.column_names = [column]
        self.features = {column: object()}
        self.types = {column: column_type} if column_type is not None else {}
        self._column = column
        self._values = list(values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, key):
        if isinstance(key, int):
            return {self._column: self._values[key]}
        if key == self._column:
            return list(self._values)
        raise KeyError(key)


class DeviceTrackingTensor:
    def __init__(self, tensor):
        self.tensor = tensor
        self.devices = []

    def to(self, device):
        self.devices.append(torch.device(device))
        return self.tensor.to(device)


class FakeProcessor:
    def __init__(self):
        self.image_batch_sizes = []
        self.image_encodings = []
        self.image_tensors = []
        self.image_batches = []

    def __call__(self, *, images=None, text=None, return_tensors="pt", padding=False):
        if images is not None:
            self.image_batch_sizes.append(len(images))
            self.image_batches.append(images)
            rows = []
            for image in images:
                red, _green, blue = image.getpixel((0, 0))
                rows.append([float(red > blue), float(blue > red)])
            pixel_values = DeviceTrackingTensor(torch.tensor(rows))
            self.image_tensors.append(pixel_values)
            encoding = BatchEncoding({"pixel_values": pixel_values})
            self.image_encodings.append(encoding)
            return encoding
        return BatchEncoding({"input_ids": torch.arange(len(text)).reshape(-1, 1)})


class FakeCLIPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))

    def get_image_features(self, pixel_values):
        return pixel_values

    def get_text_features(self, input_ids):
        return torch.eye(len(input_ids), 2)


class FakeCLIPModelWithPooling(FakeCLIPModel):
    """Mirrors transformers>=5, where get_*_features() returns a
    BaseModelOutputWithPooling instead of a plain tensor."""

    def get_image_features(self, pixel_values):
        return SimpleNamespace(pooler_output=super().get_image_features(pixel_values))

    def get_text_features(self, input_ids):
        return SimpleNamespace(pooler_output=super().get_text_features(input_ids))


@pytest.fixture
def categorical_type():
    return SimpleNamespace(categories=["dog", "cat"])


@pytest.fixture
def fake_backend():
    return SimpleNamespace(
        model=FakeCLIPModel(),
        processor=FakeProcessor(),
        text_features=torch.eye(2),
    )


@pytest.fixture
def fake_backend_with_pooling():
    return SimpleNamespace(
        model=FakeCLIPModelWithPooling(),
        processor=FakeProcessor(),
        text_features=torch.eye(2),
    )


@pytest.fixture
def fake_two_class_dashai_dataset():
    categorical = SimpleNamespace(categories=["red", "blue"])
    x = FakeDataset("image", [FakeImage("red"), FakeImage("blue")])
    y = FakeDataset("label", ["red", "blue"], categorical)
    return x, y
