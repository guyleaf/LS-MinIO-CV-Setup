from typing import Callable, Optional

import torch
from torch.utils.data import Dataset

from .common import ImageDataset

WEATHER_CLASSES = ["clear", "cloudy", "foggy", "rainy", "snowy"]
WEATHER_CLASS_TO_ID = {class_: id for id, class_ in enumerate(WEATHER_CLASSES)}


class WeatherImagesDataset(Dataset[torch.Tensor]):
    def __init__(self, files: list[str], transforms: Optional[Callable] = None) -> None:
        self.transforms = transforms
        self.dataset = ImageDataset(files=files)

    def __getitem__(self, index: int) -> torch.Tensor:
        image, _, _ = self.dataset[index]
        if self.transforms is not None:
            image = self.transforms(image)
        return image

    def __len__(self) -> int:
        return self.dataset.__len__()
