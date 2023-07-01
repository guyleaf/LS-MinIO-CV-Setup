import numpy as np
import torch
from rich.progress import track
from torch.utils.data import DataLoader

from ..datasets.weather import (
    WEATHER_CLASS_TO_ID,
    WEATHER_CLASSES,
    WeatherImagesDataset,
)
from ..models.weather import WeatherModel


@torch.no_grad()
def predict_weathers(root_dir: str, ckpt_path: str) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cpu")
    dataset = WeatherImagesDataset(root_dir)
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=4
    )

    model = WeatherModel(ckpt_path)
    model.eval()
    model = model.to(device)

    batch_y_hats = []
    for images in track(dataloader, description="Predicting...", total=len(dataloader)):
        images: torch.Tensor
        images = images.to(device)
        y_hats = model(images)

        batch_y_hats.append(y_hats.cpu())
    y_hats = torch.concatenate(batch_y_hats, dim=0)
    predictions = torch.argmax(y_hats, dim=1)

    return (
        predictions.numpy(),
        y_hats.numpy(),
    )


def collect_weather_from_tasks(
    tasks: list[dict],
) -> np.ndarray:
    weathers = []
    for task in tasks:
        for annotation in task["annotations"][0]["result"]:
            if annotation["from_name"] == "weather":
                weather = annotation["value"]["choices"][0]
                weathers.append(WEATHER_CLASS_TO_ID[weather])
                break

    return np.array(weathers)


def get_weather_classes() -> list[str]:
    return WEATHER_CLASSES
