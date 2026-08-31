from types import SimpleNamespace

import pytest
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


@pytest.fixture
def categorical_type():
    return SimpleNamespace(categories=["dog", "cat"])
