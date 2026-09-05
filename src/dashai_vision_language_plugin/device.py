import torch


def resolve_device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, or cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use device='auto' or 'cpu'")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
