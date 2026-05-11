"""ImageInput: wraps a PIL Image or file path and converts to a tensor."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import torch


class ImageInput:
    """Holds an image (path or PIL Image) and converts it to a (C, H, W) tensor."""

    def __init__(self, source: Union[str, Path, Any]) -> None:
        self._source = source
        self._tensor: torch.Tensor | None = None

    @property
    def raw(self) -> Any:
        return self._source

    def to_tensor(self) -> torch.Tensor:
        if self._tensor is not None:
            return self._tensor
        try:
            from PIL import Image
            import torchvision.transforms.functional as TF
        except ImportError as exc:
            raise ImportError("ImageInput needs Pillow and torchvision.") from exc

        if isinstance(self._source, (str, Path)):
            img = Image.open(self._source).convert("RGB")
        else:
            img = self._source.convert("RGB")

        self._tensor = TF.to_tensor(img)
        return self._tensor
