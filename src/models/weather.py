import torch
import torch.nn as nn
from uav_research.models.backbones import EfficientNet
from uav_research.models.weather import WeatherModel as UAVWeatherModel
from uav_research.modules.tasks import ClassificationTask

from ..datasets.weather import WEATHER_CLASSES


class WeatherModel(nn.Module):
    def __init__(self, ckpt_path: str) -> None:
        super().__init__()

        # TODO: refactor this
        self.model = ClassificationTask.load_from_checkpoint(
            ckpt_path,
            model=UAVWeatherModel(
                len(WEATHER_CLASSES),
                EfficientNet(
                    "efficientnet_v2_s", "EfficientNet_V2_S_Weights.IMAGENET1K_V1"
                ),
                backbone_out_features=1280,
                dropout=0.2,
            ),
        )
        self.model.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)
