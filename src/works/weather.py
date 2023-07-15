import clip
import numpy as np
import torch
from rich.progress import track
from torch.utils.data import DataLoader
from uav_research.transforms.weather import WeatherPreprocessTransform

from ..datasets.weather import (
    WEATHER_CLASS_TO_ID,
    WEATHER_CLASSES,
    WeatherImagesDataset,
)
from ..models.weather import WeatherModel

CLIP_MODEL = "ViT-B/16"


def predict_weathers_by_clip(
    files: list[str], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    tokens = clip.tokenize(
        [f"The weather is {class_name.lower()}." for class_name in WEATHER_CLASSES]
    ).to(device)
    dataset = WeatherImagesDataset(files, transforms=preprocess)
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=4
    )

    batch_y_hats = []
    for images in track(dataloader, description="Predicting...", total=len(dataloader)):
        images: torch.Tensor
        images = images.to(device, non_blocking=True)
        logits, _ = model(images, tokens)
        y_hats = torch.softmax(logits, -1)

        batch_y_hats.append(y_hats.cpu())
    y_hats = torch.concatenate(batch_y_hats, dim=0)
    predictions = torch.argmax(y_hats, dim=1)

    return (
        predictions,
        y_hats,
    )


def predict_weathers_by_weather(
    files: list[str], ckpt_path: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model = WeatherModel(ckpt_path)
    model.eval()
    model = model.to(device)

    dataset = WeatherImagesDataset(
        files, transforms=WeatherPreprocessTransform(crop_size=384, train=False)
    )
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=4
    )

    batch_y_hats = []
    for images in track(dataloader, description="Predicting...", total=len(dataloader)):
        images: torch.Tensor
        images = images.to(device, non_blocking=True)
        y_hats = model(images)

        batch_y_hats.append(y_hats.cpu())
    y_hats = torch.concatenate(batch_y_hats, dim=0)
    predictions = torch.argmax(y_hats, dim=1)

    return (
        predictions,
        y_hats,
    )


@torch.no_grad()
def predict_weathers_ensemble(
    files: list[str], ckpt_path: str
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda")

    weather_predictions, weather_y_hats = predict_weathers_by_weather(
        files, ckpt_path, device
    )
    clip_predictions, clip_y_hats = predict_weathers_by_clip(files, device)

    y_hats = torch.stack([weather_y_hats, clip_y_hats], dim=0)
    predictions = torch.stack([weather_predictions, clip_predictions], dim=0)

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
