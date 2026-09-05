import pytest
import torch

from dashai_vision_language_plugin.device import resolve_device


def test_auto_uses_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")


def test_cpu_is_explicit():
    assert resolve_device("cpu") == torch.device("cpu")


def test_cuda_unavailable_is_clear(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_device("cuda")


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="auto, cpu, or cuda"):
        resolve_device("mps")
