import torch
from torch.utils.data import Dataset
from uav_research.transforms.weather import WeatherPreprocessTransform

from ..datasets.common import ImageDataset

WEATHER_CLASSES = ["clear", "cloudy", "foggy", "rainy", "snowy"]
WEATHER_CLASS_TO_ID = {class_: id for id, class_ in enumerate(WEATHER_CLASSES)}


class WeatherImagesDataset(Dataset[torch.Tensor]):
    def __init__(self, root_dir: str) -> None:
        self.transforms = WeatherPreprocessTransform(crop_size=384, train=False)
        self.dataset = ImageDataset(root_dir)

    def __getitem__(self, index: int) -> torch.Tensor:
        image, _, _ = self.dataset[index]
        image = self.transforms(image)
        return image

    def __len__(self) -> int:
        return self.dataset.__len__()
