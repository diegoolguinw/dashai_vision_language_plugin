from types import SimpleNamespace

import pytest
import torch
from PIL import Image


class FakeImage:
    def __init__(self, color):
        self.color = color

    def to_pil(self):
        return Image.new("RGB", (2, 2), self.color)


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


class FakeProcessor:
    def __init__(self):
        self.image_batch_sizes = []

    def __call__(self, *, images=None, text=None, return_tensors="pt", padding=False):
        if images is not None:
            self.image_batch_sizes.append(len(images))
            rows = []
            for image in images:
                red, _green, blue = image.getpixel((0, 0))
                rows.append([float(red > blue), float(blue > red)])
            return {"pixel_values": torch.tensor(rows)}
        return {"input_ids": torch.arange(len(text)).reshape(-1, 1)}


class FakeCLIPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))

    def get_image_features(self, pixel_values):
        return pixel_values

    def get_text_features(self, input_ids):
        return torch.eye(len(input_ids), 2)


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
