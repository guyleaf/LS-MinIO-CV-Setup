import clip
import numpy as np
import skimage.color as skcolor
import torch
from rich.progress import track
from torch.utils.data import DataLoader

from ..datasets.intensity import (
    INTENSITY_CLASS_TO_ID,
    INTENSITY_CLASSES,
    IntensityImagesDataset,
)
from ..utils import convert_annotations_to_probabilities

CLIP_MODEL = "ViT-B/16"


@torch.no_grad()
def predict_intensity_by_clip(
    root_dir: str, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    tokens = clip.tokenize(
        [
            f"The overall brightness level is {class_name.lower()}."
            for class_name in INTENSITY_CLASSES
        ]
    ).to(device)
    dataset = IntensityImagesDataset(root_dir, transforms=preprocess)
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
        predictions.numpy(),
        y_hats.numpy(),
    )


def measure_intensity_level(
    root_dir: str, thresholds: list[float] = [0.33, 0.67]
) -> np.ndarray:
    if len(thresholds) != 2:
        raise ValueError("The thresholds should only contain two floating values")

    levels = []
    dataset = IntensityImagesDataset(root_dir)
    for rgb in dataset:
        lab = skcolor.rgb2lab(rgb)
        intensity = lab[..., 0] / 100
        average_intensity = np.mean(intensity)

        level = "medium"
        if average_intensity >= thresholds[1]:
            level = "bright"
        elif average_intensity <= thresholds[0]:
            level = "dark"
        levels.append(INTENSITY_CLASS_TO_ID[level])

    return np.array(levels)


def predict_intensity_ensemble(
    root_dir: str, thresholds: list[float] = [0.33, 0.67]
) -> tuple[np.ndarray, np.ndarray]:
    intensity_levels = measure_intensity_level(root_dir, thresholds=thresholds)
    intensity_hats = convert_annotations_to_probabilities(
        intensity_levels, len(get_intensity_classes())
    )

    device = torch.device("cuda")
    clip_intensity_predictions, clip_intensity_hats = predict_intensity_by_clip(
        root_dir, device
    )

    intensity_levels = np.stack([intensity_levels, clip_intensity_predictions], axis=0)
    intensity_hats = np.stack([intensity_hats, clip_intensity_hats], axis=0)
    return intensity_levels, intensity_hats


def collect_intensity_level_from_tasks(
    tasks: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    intensities_1 = []
    intensities_2 = []
    for task in tasks:
        for annotation in task["annotations"][0]["result"]:
            if annotation["from_name"] == "intensity_1":
                level = annotation["value"]["choices"][0]
                intensities_1.append(INTENSITY_CLASS_TO_ID[level])
            elif annotation["from_name"] == "intensity_2":
                level = annotation["value"]["choices"][0]
                intensities_2.append(INTENSITY_CLASS_TO_ID[level])

    return np.array(intensities_1), np.array(intensities_2)


def get_intensity_classes() -> list[str]:
    return INTENSITY_CLASSES
